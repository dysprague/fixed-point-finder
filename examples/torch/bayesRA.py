'''
examples/torch/FlipFlop.py
Written for Python 3.8.17 and Pytorch 2.0.1
@ Matt Golub, June 2023
Please direct correspondence to mgolub@cs.washington.edu
'''

import pdb

import sys
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from scipy.special import i0, i1
from scipy.optimize import root_scalar

PATH_TO_FIXED_POINT_FINDER = '../../'
sys.path.insert(0, PATH_TO_FIXED_POINT_FINDER)
from FixedPointFinderTorch import FixedPointFinderTorch as FixedPointFinder
from FixedPoints import FixedPoints

from FlipFlopData import FlipFlopData

def A_Bessel(kappa): 
	"""Computes the ratio of Bessel functions."""
	r = i1(kappa)/i0(kappa)
	return r

def xi_fun_inv(dt):
	"""Computes the inverse of the ratio of Bessel functions by root-finding."""
	f = lambda alpha: alpha * A_Bessel(alpha) - dt
	sol = root_scalar(f,bracket=[0.001,50],method='brentq')
	alpha = sol.root
	return alpha

def update_r_t_batched(r_t_minus_1, M, W,
					   dt, alpha, kappa_phi, kappa_v, kappa_z,
					   phi_0_r, HD_input):
	"""
	Batched version of your update using PyTorch.

	Parameters
	----------
	r_t_minus_1 : torch.Tensor, shape (1, b, N)
	M           : torch.Tensor, shape (N, N, b)
	W           : torch.Tensor, shape (N, N, b)
	dt, alpha, kappa_phi, kappa_v, kappa_z : floats (scalars)
	phi_0_r, HD_input : torch.Tensor, shape (1, b, N) or broadcastable to that

	Returns
	-------
	r_t : torch.Tensor, shape (1, b, N)
	"""

	# --- 1) Compute sqrt( r_{t-1}^T M r_{t-1} ) per batch => shape (1, b)
	# Using torch.einsum for batched operations
	# r_t_minus_1: (1, b, N)
	# M: (N, N, b)
	# We need to align the batch dimension correctly
	# Rearrange M to (b, N, N) for easier computation
	M = M.permute(2, 0, 1)  # (b, N, N)
	r = r_t_minus_1.squeeze(0)  # (b, N)

	# Compute r^T M r for each batch
	# First compute M r -> (b, N)
	Mr = torch.bmm(M, r.unsqueeze(2)).squeeze(2)  # (b, N)

	# Then compute element-wise r * (M r) and sum over N to get (b,)
	rMr = torch.sum(r * Mr, dim=1)  # (b,)

	# Take square root
	val_sqrt = torch.sqrt(rMr).unsqueeze(0)  # (1, b)

	# --- 2) Compute W @ r_{t-1} per batch => shape (1, b, N)
	# Rearrange W to (b, N, N)
	W = W.permute(2, 0, 1)  # (b, N, N)

	# Compute W r -> (b, N)
	Wr = torch.bmm(W, r.unsqueeze(2)).squeeze(2)  # (b, N)

	# Reshape to (1, b, N)
	Wr = Wr.unsqueeze(0)  # (1, b, N)

	# --- 3) Combine terms (element-wise + broadcasting)
	decay_term = alpha * r_t_minus_1 * dt  # (1, b, N)

	# val_sqrt: (1, b), need to unsqueeze to (1, b, 1) for broadcasting
	nonlinear_term = (1.0 / (kappa_phi + kappa_v)) * val_sqrt.unsqueeze(2) * r_t_minus_1 * dt  # (1, b, N)

	recurrent_term = Wr * dt  # (1, b, N)

	# Ensure phi_0_r and HD_input are (1, b, N)
	input_term = kappa_z * torch.cos(phi_0_r - HD_input)  # (1, b, N)

	# Update r_t
	r_t = (r_t_minus_1
		   - decay_term
		   - nonlinear_term
		   + recurrent_term
		   + input_term)

	return r_t

class BayesRingRNN(nn.Module):
	
	def __init__(self, num_units=80, dt=0.01, kappa_phi=1, kappa_v=2, kappa_z=15, batch_size=1024, batch_first=True, device='cpu'):
		super().__init__()

		np.random.seed(42)
		self.T = 15
		self.dt = dt
		self.t = np.arange(0,self.T,self.dt)
		self.kappa_phi = kappa_phi # inverse diffusion constant
		self.kappa_v = kappa_v # precision of relative heading info
		#self.kappa_z = kappa_z # precision of absolute heading info (called gamma_z in manuscript)
		self.alpha = xi_fun_inv(kappa_z*self.dt) 
		self.batch_first = batch_first
		
		self.kappa_z = self.alpha

		self.N = num_units

		self.phi_0_r = torch.from_numpy(np.linspace(-np.pi,np.pi-(2*np.pi)/self.N,self.N))
		self.alpha = 100
		
		##### connectivity matrices
		self.W_rec_even = np.zeros((self.N,self.N, batch_size)) # NxNxb
		for i in range(self.N):
			for j in range(self.N):
				for k in range(batch_size):
					self.W_rec_even[i,j,k] = 2/self.N * np.cos(self.phi_0_r[i] - self.phi_0_r[j])

		self.W_rec_even = torch.from_numpy(self.W_rec_even).to(torch.float64)
				
		self.W_rec_odd = np.zeros((self.N,self.N, batch_size))
		for i in range(self.N):
			for j in range(self.N):
				for k in range(batch_size):
					self.W_rec_odd[i,j,k] = 2/self.N * np.sin(self.phi_0_r[i] - self.phi_0_r[j])

		self.W_rec_odd = torch.from_numpy(self.W_rec_odd).to(torch.float64)
				
		self.M_inh = np.zeros((self.N,self.N, batch_size)) # NxNxb
		for i in range(self.N):
			for j in range(self.N):
				for k in range(batch_size):
					self.M_inh[i,j] =  (2/self.N)**2 * np.cos(self.phi_0_r[i] - self.phi_0_r[j])

		self.M_inh = torch.from_numpy(self.M_inh).to(torch.float64)
		
	def forward(self, inputs, r0=None): # input with shape batch, time, num inputs; r0 with shape 1, batch, num neurons
		output = []

		if self.batch_first:
			inputs = inputs.transpose(0,1) # now time, batch, num_inputs

		seq_len, batch_size, _ = inputs.shape

		if self.W_rec_even.shape[2] != batch_size:
			##### connectivity matrices
			self.W_rec_even = np.zeros((self.N,self.N, batch_size)) # NxNxb
			for i in range(self.N):
				for j in range(self.N):
					for k in range(batch_size):
						self.W_rec_even[i,j,k] = 2/self.N * np.cos(self.phi_0_r[i] - self.phi_0_r[j])

			self.W_rec_even = torch.from_numpy(self.W_rec_even).to(torch.float64)
					
			self.W_rec_odd = np.zeros((self.N,self.N, batch_size))
			for i in range(self.N):
				for j in range(self.N):
					for k in range(batch_size):
						self.W_rec_odd[i,j,k] = 2/self.N * np.sin(self.phi_0_r[i] - self.phi_0_r[j])

			self.W_rec_odd = torch.from_numpy(self.W_rec_odd).to(torch.float64)
					
			self.M_inh = np.zeros((self.N,self.N, batch_size)) # NxNxb
			for i in range(self.N):
				for j in range(self.N):
					for k in range(batch_size):
						self.M_inh[i,j] =  (2/self.N)**2 * np.cos(self.phi_0_r[i] - self.phi_0_r[j])

			self.M_inh = torch.from_numpy(self.M_inh).to(torch.float64)
		
		HD_input = inputs[:,:,0] # time, batch, val
		AV_input = inputs[:,:,1] 

		HD_input = HD_input[:,:,np.newaxis] #txbx1
		AV_input = AV_input[:,np.newaxis,:] #tx1xb
		
		phi_0_r = torch.from_numpy(np.linspace(-np.pi,np.pi-(2*np.pi)/self.N,self.N))

		phi_0_r_batch = phi_0_r[np.newaxis, np.newaxis, :]  # shape (1, 1, n)
		phi_0_r_batch = np.repeat(phi_0_r_batch, batch_size, axis=1)    # shape (1, b, n)
		alpha = 100
				
		# set network parameters to match filtering
		alpha_tilde = 0
		alpha = alpha_tilde + 1/2 * self.kappa_v/self.kappa_phi * 1/(self.kappa_phi + self.kappa_v)
		W = ( (alpha_tilde + 1/(self.kappa_phi + self.kappa_v)) * self.W_rec_even 
			+ (self.kappa_v/(self.kappa_v+self.kappa_phi))*self.W_rec_odd * AV_input[0].unsqueeze(0)/self.dt )
		M = self.M_inh
		
		kappa_0 = 10 # initial certainty
		phi_0 = 0 # initial head direction
		# init
		if r0 is None:
			r_t_minus_1 = kappa_0 * torch.from_numpy(np.cos(phi_0_r_batch - phi_0)).to(torch.float64)
			r_t = kappa_0 * torch.from_numpy(np.cos(phi_0_r_batch - phi_0)).to(torch.float64)
		
		else:
			r0 = r0.to(torch.float64)
			r_t_minus_1 = r0
			r_t = r0

		# run network filter
		for i in range(seq_len):
			W =  ( (alpha_tilde + 1/(self.kappa_phi + self.kappa_v)) *self. W_rec_even 
			+ (self.kappa_v/(self.kappa_v+self.kappa_phi))*self.W_rec_odd * AV_input[i].unsqueeze(0)/self.dt )
			
			r_t = update_r_t_batched(r_t_minus_1, M, W,
					   self.dt, alpha, self.kappa_phi, self.kappa_v, self.kappa_z,
					   phi_0_r_batch, HD_input[i].unsqueeze(0))
			
			#r_t = (r_t_minus_1 
			#		- alpha * r_t_minus_1 * self.dt # decay
			#		- 1/(self.kappa_phi+self.kappa_v) * np.sqrt(np.einsum("abc, cdb, abd -> ab",r_t_minus_1, M, r_t_minus_1))[:, :, np.newaxis] * r_t_minus_1 * self.dt # coincidence detector
			#		+ np.einsum("ijb, 1bj -> 1bi", W, r_t_minus_1) * self.dt # angular velocity integration
			#	+ self.kappa_z*np.cos(phi_0_r_batch-HD_input)) # absolute heading info (external input)
			
			output.append(r_t) #1xbxd
			r_t_minus_1 = r_t
			
		output = torch.stack(output).squeeze()

		if output.dim()==2:
			output = output.unsqueeze(0)

		if output.dim()==1:
			output = output.unsqueeze(0).unsqueeze(0)

		if self.batch_first:
				output = output.transpose(0,1)
		
		return output, r_t
		