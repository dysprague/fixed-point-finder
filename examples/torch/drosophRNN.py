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

import torch.nn.functional as F

PATH_TO_FIXED_POINT_FINDER = '../../'
sys.path.insert(0, PATH_TO_FIXED_POINT_FINDER)
from FixedPointFinderTorch import FixedPointFinderTorch as FixedPointFinder
from FixedPoints import FixedPoints

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

def IdMatr(x_to,x_from,sig=0.001):
	N_to = x_to.size
	N_from = x_from.size
	IdM = np.zeros((N_to,N_from))
	for i in range(N_to):
		for j in range(N_from):
			dist = np.arccos(np.cos(x_to[i]-x_from[j]))
			IdM[i,j] = (1/(np.sqrt(2*np.pi)*sig)
							* np.exp(-(1/2)*(dist**2/sig**2 ) ) )
	IdM = IdM/np.sum(IdM,1)[0]  # normalize
	return IdM

def fact(x):

	return F.relu(x)

def create_connmatrix(N_HD,N_Del7,N_AVplus,N_AVminus):
	
	sig_AVminus = 0.3
	sig_AVplus = 0.3
	sig_Del7 = 0.1

	# preferred angles
	phi_0_HD = np.linspace(-np.pi,np.pi-(2*np.pi)/N_HD,N_HD)  # HD preferred angle
	phi_0_Del7 = np.linspace(-np.pi,np.pi-(2*np.pi)/N_Del7,N_Del7)
	phi_0_AVplus = np.linspace(-np.pi,np.pi-(2*np.pi)/N_AVplus,N_AVplus)
	phi_0_AVminus = np.linspace(-np.pi,np.pi-(2*np.pi)/N_AVminus,N_AVminus)

	### HD population
	phi_0_HD = np.linspace(-np.pi,np.pi-(2*np.pi)/N_HD,N_HD)  # preferred angle

	W_HD_HD = np.zeros((N_HD,N_HD)) # recurrent connectivity matrix HD -> HD
	w0 = 0
	w1 = 2
	o = -0.2
	for i in range(N_HD):
		for j in range(N_HD):
			W_HD_HD[i,j] = 2/N_HD * (o + np.maximum(0,w0+w1*np.cos(phi_0_HD[i]-phi_0_HD[j]))) 
			
	W_HD_AVplus = np.zeros((N_HD,N_AVplus))
	for i in range(N_HD):
		for j in range(N_AVplus):
			W_HD_AVplus[i,j] = 2/N_AVplus * np.maximum(0,np.sin(phi_0_HD[i]-phi_0_AVplus[j]+np.pi/4))
			
	W_HD_AVminus = np.zeros((N_HD,N_AVminus))
	for i in range(N_HD):
		for j in range(N_AVminus):
			W_HD_AVminus[i,j] = 2/N_AVminus * np.maximum(0,-np.sin(phi_0_HD[i]-phi_0_AVminus[j]-np.pi/4))


	## AVplus population
	W_AVplus_HD = IdMatr(phi_0_AVplus,phi_0_HD,sig_AVplus) 

	## AVminus population
	W_AVminus_HD = IdMatr(phi_0_AVminus,phi_0_HD,sig_AVminus) 

	## Delta7 population
	W_Del7_HD = np.zeros((N_Del7,N_HD)) # HD -> GI connectivity matrix
	w0 = 0.5 #(np.pi-1)/np.pi
	w1 = - 1/2
	for i in range(N_Del7):
		for j in range(N_HD):
			W_Del7_HD[i,j] = 2/N_Del7 *(w0 + w1* np.cos(phi_0_Del7[i] - phi_0_HD[j])) 
			# W_Del7_HD[i,j] = 2/N_HD *(w0 + w1* np.cos(phi_0_Del7[i] - phi_0_HD[j]))

	m0 = 0.1
	m1 = 0
	W_Del7_Del7 = np.zeros((N_Del7,N_Del7))
	for i in range(N_Del7):
		for j in range(N_Del7):
			W_Del7_Del7[i,j] = 2/N_Del7 *(m0 + m1* np.cos(phi_0_Del7[i] - phi_0_Del7[j]))


	j = np.pi/2 * (1 - 2*m0)/w0
	W_HD_Del7 = - j * IdMatr(phi_0_HD,phi_0_Del7,sig_Del7)  

	W_HD_HD = torch.from_numpy(W_HD_HD).to(torch.float64)
	W_HD_AVplus = torch.from_numpy(W_HD_AVplus).to(torch.float64)
	W_HD_AVminus = torch.from_numpy(W_HD_AVminus).to(torch.float64)
	W_AVplus_HD = torch.from_numpy(W_AVplus_HD).to(torch.float64)
	W_AVminus_HD = torch.from_numpy(W_AVminus_HD).to(torch.float64)
	W_Del7_HD = torch.from_numpy(W_Del7_HD).to(torch.float64)
	W_Del7_Del7 = torch.from_numpy(W_Del7_Del7).to(torch.float64)
	W_HD_Del7 = torch.from_numpy(W_HD_Del7).to(torch.float64) 

	params = {
		"W_HD_HD" : W_HD_HD,
		"W_HD_AVplus" : W_HD_AVplus,
		"W_HD_AVminus" : W_HD_AVminus,
		"W_AVplus_HD" : W_AVplus_HD,
		"W_AVminus_HD" : W_AVminus_HD,
		"W_Del7_HD" : W_Del7_HD,
		"W_Del7_Del7" : W_Del7_Del7,
		"W_HD_Del7" : W_HD_Del7
	}

	return params


def update_r_HD_batched(r_HD_minus_1, r_AVplus_minus_1, r_AVminus_minus_1, r_Del7_minus_1, I_ext, w_HD_HD,
						W_HD_HD, w_HD_AVplus, W_HD_AVplus, w_HD_AVminus, W_HD_AVminus, w_HD_Del7, W_HD_Del7, dt, alpha, phi_0_r, HD_input):
	
	leak_term = alpha* r_HD_minus_1*dt
	recurr_term = w_HD_HD * torch.einsum('ij,bj->bi', W_HD_HD, r_HD_minus_1)*dt
	AVplus_term = w_HD_AVplus * torch.einsum('ij,bj->bi', W_HD_AVplus, r_AVplus_minus_1) * dt
	AVminus_term = w_HD_AVminus * torch.einsum('ij,bj->bi', W_HD_AVminus, r_AVminus_minus_1) * dt
	quad_term = w_HD_Del7 * torch.einsum('ij,bj->bi', W_HD_Del7, fact(r_Del7_minus_1)) * r_HD_minus_1 * dt
	input_term = I_ext * torch.cos(phi_0_r[None,:] - HD_input.transpose(0,1))
	
	r_HD = (r_HD_minus_1 
		 	- leak_term
			+ recurr_term
			+ AVplus_term 
			+ AVminus_term
			+ quad_term)
			#+ input_term)

	return r_HD

def update_r_AV_batched(r_AV_minus_1, tau_AV, dy, AV_offset, w_AV_HD, W_AV_HD, r_HD, dt):

	r_AV = r_AV_minus_1 + 1/tau_AV * (-r_AV_minus_1 + (dy/dt + AV_offset)[:,None] * w_AV_HD * torch.einsum('ij,bj->bi', W_AV_HD, r_HD))*dt

	return r_AV

def update_r_Del7_batched(r_Del7_minus_1, tau_Del7, w_Del7_HD, W_Del7_HD, r_HD, W_Del7_Del7, dt):

	r_Del7 = (r_Del7_minus_1
                + 1/tau_Del7 * (-r_Del7_minus_1 
                                + w_Del7_HD * torch.einsum('ij,bj->bi',W_Del7_HD, fact(r_HD)) 
                                + torch.einsum('ij,bj->bi', W_Del7_Del7, fact(r_Del7_minus_1)) )  * dt )

	return r_Del7


class drosophRNN(nn.Module):
	
	def __init__(self, dt=0.001, kappa_phi=1, kappa_y=1, kappa_z=15, AV_offset=0, batch_size=1024, batch_first=True, device='cpu'):
		super().__init__()

		#np.random.seed(42)
		self.T = 20
		self.dt = dt
		self.t = np.arange(0,self.T,self.dt)
		self.kappa_phi = kappa_phi # inverse diffusion constant
		self.kappa_y = kappa_y # precision of relative heading info
		self.kappa_z = kappa_z # precision of absolute heading info (called gamma_z in manuscript)
		self.alpha = xi_fun_inv(self.kappa_z *dt)
		self.alpha_tilde = 10
		
		self.batch_first = batch_first

		self.N_HD = 100
		self.N_Del7 = 100
		self.N_AVplus = 50
		self.N_AVminus = 50

		self.phi_0_r_HD = torch.from_numpy(np.linspace(-np.pi,np.pi-(2*np.pi)/self.N_HD,self.N_HD))
		self.phi_0_r_Del7 = torch.from_numpy(np.linspace(-np.pi,np.pi-(2*np.pi)/self.N_Del7,self.N_Del7))

		params = create_connmatrix(self.N_HD,self.N_Del7,self.N_AVplus,self.N_AVminus) #TODO: update to be batched

		#unpack
		self.W_HD_HD = params['W_HD_HD']
		self.W_HD_AVplus = params['W_HD_AVplus']
		self.W_HD_AVminus = params['W_HD_AVminus']
		self.W_AVplus_HD = params['W_AVplus_HD']
		self.W_AVminus_HD = params['W_AVminus_HD']
		self.W_Del7_Del7 = params['W_Del7_Del7']
		self.W_HD_Del7 = params['W_HD_Del7']
		self.W_Del7_HD = params['W_Del7_HD']

		self.w0 = 0 
		self.w1 = 2
		self.m0 = 0.1
		self.m1 = 0

		self.AV_offset = AV_offset

		# compute the prefactors (strength modulator)
		self.w_HD_HD = self.alpha_tilde + 1/(self.kappa_phi + self.kappa_y) - self.AV_offset*self.kappa_y/(self.kappa_y+self.kappa_phi)
		self.w_HD_AVplus = np.sqrt(2) * self.kappa_y/(self.kappa_y+self.kappa_phi) # note sqrt(2) due to 45deg
		self.w_HD_AVminus = np.sqrt(2) * self.kappa_y/(self.kappa_y+self.kappa_phi)
		self.w_HD_Del7 = 1/(self.kappa_phi+self.kappa_y) 
		self.w_AVplus_HD = 1
		self.w_AVminus_HD = 1
		self.w_Del7_HD = 1

		# time constants
		self.tau_AVplus = 0.01
		self.tau_AVminus = 0.01
		self.tau_Del7 = 0.001

		self.kappa_0 = 10
		self.phi_0 = 0
		
	def forward(self, inputs, r0): # input with shape batch, time, num inputs; r0 with shape 1, batch, num neurons
		output = []

		if self.batch_first:
			inputs = inputs.transpose(0,1) # now time, batch, num_inputs

		seq_len, batch_size, _ = inputs.shape
		
		HD_input = inputs[:,:,0] # time, batch, val
		AV_input = inputs[:,:,1] 

		HD_input = HD_input[:,:] #txb
		AV_input = AV_input[:,:] #txb
		
		# init
		#TODO: separate  input into different subsections of neurons
		r0 = r0.to(torch.float64)

		phi_0_r = torch.from_numpy(np.linspace(-np.pi,np.pi-(2*np.pi)/self.N_HD,self.N_HD))

		r_HD_minus_1 = r0[0,:,:self.N_HD]
		r_AVplus_minus_1 = r0[0,:,self.N_HD:self.N_HD+self.N_AVplus]
		r_AVminus_minus_1 = r0[0,:,self.N_HD+self.N_AVplus:self.N_HD+self.N_AVplus+self.N_AVminus]
		r_Del7_minus_1 = r0[0,:,self.N_HD+self.N_AVplus+self.N_AVminus:]

		alpha = self.alpha_tilde + 1/2 * self.kappa_y/self.kappa_phi * 1/(self.kappa_phi + self.kappa_y)
		I_ext = self.alpha


		# run network filter
		for i in range(seq_len):
			
			r_HD = update_r_HD_batched(r_HD_minus_1, r_AVplus_minus_1, r_AVminus_minus_1, r_Del7_minus_1, I_ext, 
						self.w_HD_HD, self.W_HD_HD, self.w_HD_AVplus, self.W_HD_AVplus, self.w_HD_AVminus, self.W_HD_AVminus,
						self.w_HD_Del7, self.W_HD_Del7, self.dt, alpha, phi_0_r, HD_input[i].unsqueeze(0))
			
			r_AVplus = update_r_AV_batched(r_AVplus_minus_1, self.tau_AVplus, AV_input[i], self.AV_offset, self.w_AVplus_HD, self.W_AVplus_HD, r_HD, self.dt)
			r_AVminus = update_r_AV_batched(r_AVminus_minus_1, self.tau_AVminus, -AV_input[i], self.AV_offset, self.w_AVminus_HD, self.W_AVminus_HD, r_HD, self.dt)
	
			r_Del7 = update_r_Del7_batched(r_Del7_minus_1, self.tau_Del7, self.w_Del7_HD, self.W_Del7_HD, r_HD, self.W_Del7_Del7, self.dt)

			r_full_curr = torch.cat((r_HD, r_AVplus, r_AVminus, r_Del7), axis=1)
			
			output.append(r_full_curr) #1xbxd #TODO: stack outputs for each subsection of neurons

			r_HD_minus_1 = r_HD 
			r_AVplus_minus_1 = r_AVplus 
			r_AVminus_minus_1 = r_AVminus 
			r_Del7_minus_1 = r_Del7
			
		output = torch.stack(output).squeeze()

		if output.dim()==2:
			output = output.unsqueeze(0)

		if output.dim()==1:
			output = output.unsqueeze(0).unsqueeze(0)

		if self.batch_first:
				output = output.transpose(0,1)
		
		return output, r_full_curr
		