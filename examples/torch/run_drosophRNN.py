'''
examples/torch/run_FlipFlop.py
Written for Python 3.8.17 and Pytorch 2.0.1
@ Matt Golub, June 2023
Please direct correspondence to mgolub@cs.washington.edu
'''

import pdb
import sys
import numpy as np

PATH_TO_FIXED_POINT_FINDER = '../../'
PATH_TO_HELPER = '../helper/'
sys.path.insert(0, PATH_TO_FIXED_POINT_FINDER)
sys.path.insert(0, PATH_TO_HELPER)

from FlipFlop import FlipFlop
from FixedPointFinderTorch import FixedPointFinderTorch as FixedPointFinder
from FlipFlopData import FlipFlopData
from plot_utils import plot_fps
from drosophRNN import drosophRNN
import matplotlib.pyplot as plt
import torch

from scipy.special import i0, i1
from scipy.optimize import root_scalar

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

def generate_data(T=5, dt=0.01,N=80, kappa_phi=1, kappa_v=2, kappa_z=15,batch_size=200):
	
	t = np.arange(0,T,dt)
	alpha = xi_fun_inv(kappa_z*dt) 

	### create kappa_z depending on environment. If light is off, kappa_z is zero (maximally uninformative)
	kappa_z_array = alpha * np.ones(int(T/dt)) # precision of absolute heading info

	trajectories = np.zeros((batch_size,len(t),N))
	true_HD = np.zeros((batch_size, len(t)))
	HD_obs = np.zeros((batch_size, len(t)))
	AV_obs = np.zeros((batch_size, len(t)))
	
	for batch in range(batch_size):
		### create hidden state trajectory and observation sequence
		phi = np.zeros(int(T/dt)) # latent state
		phi_0 = 0 
		kappa_0 = 10 # initial certainty
		phi[0] = np.random.vonmises(phi_0,kappa_0)
		phi[0] = (phi[0] + np.pi ) 
		dy = np.zeros(int(T/dt))
		for i in range(1,int(T/dt)):
				phi[i] = np.random.normal(phi[i-1],1/np.sqrt(kappa_phi) * np.sqrt(dt)) #True HD observations
				dy[i] = np.random.normal(phi[i]-phi[i-1],1/np.sqrt(kappa_v) * np.sqrt(dt)) #Angular velocity observations
		phi = (phi % (2*np.pi) ) - np.pi # convert true HD to circular coordinates
		mu_z = np.random.vonmises(phi,kappa_z_array) # draw HD observations from Von Mises distribution based on true HD

		phi_0_r = np.linspace(-np.pi,np.pi-(2*np.pi)/N,N)
		alpha = 100

		##### connectivity matrices
		W_rec_even = np.zeros((N,N))
		for i in range(N):
			for j in range(N):
				W_rec_even[i,j] = 2/N * np.cos(phi_0_r[i] - phi_0_r[j])
				
		W_rec_odd = np.zeros((N,N))
		for i in range(N):
			for j in range(N):
				W_rec_odd[i,j] = 2/N * np.sin(phi_0_r[i] - phi_0_r[j])
				
		M_inh = np.zeros((N,N))
		for i in range(N):
			for j in range(N):
				M_inh[i,j] =  (2/N)**2 * np.cos(phi_0_r[i] - phi_0_r[j])

		# set network parameters to match filtering
		alpha_tilde = 0
		alpha = alpha_tilde + 1/2 * kappa_v/kappa_phi * 1/(kappa_phi + kappa_v)
		W = ( (alpha_tilde + 1/(kappa_phi + kappa_v)) * W_rec_even 
			+ (kappa_v/(kappa_v+kappa_phi))*W_rec_odd * dy[0]/dt )
		M =  M_inh

		# init
		r = np.zeros((int(T/dt),N))
		r[0] = kappa_0 * np.cos(phi_0_r - phi_0)

		# run network filter
		for i in range(1,int(T/dt)):
			W =  ( (alpha_tilde + 1/(kappa_phi + kappa_v)) * W_rec_even 
			+ (kappa_v/(kappa_v+kappa_phi))*W_rec_odd * dy[i]/dt )
			r[i] = (r[i-1] 
					- alpha * r[i-1] * dt # decay
					- 1/(kappa_phi+kappa_v) * np.sqrt(np.dot(r[i-1],np.dot(M,r[i-1]))) * r[i-1] * dt # coincidence detector
					+ np.dot(W,r[i-1]) * dt # angular velocity integration
				+ kappa_z_array[i]*np.cos(phi_0_r-mu_z[i])) # absolute heading info (external input)
			
		trajectories[batch, :,:] = r
		true_HD[batch,:] = phi
		HD_obs[batch,:] = mu_z
		AV_obs[batch,:] = dy

	return trajectories, true_HD, HD_obs, AV_obs

def get_mu_kappa(N_HD, r_HD):
    phi_0_HD = np.linspace(-np.pi,np.pi-(2*np.pi)/N_HD,N_HD)  # HD preferred angle

    # decode stochastic variables from HD population
    A_cos =  np.array([np.cos(phi_0_HD),np.sin(phi_0_HD)])
    theta = 2/N_HD * np.einsum('LN,bN->bL', A_cos, r_HD) # FT in Cartesian domain
    kappa = np.sqrt(np.sum(theta**2,axis=-1)) #convert to polar coordinates
    mu = np.arctan2(theta[...,1],theta[...,0])
    return mu, kappa

def find_fixed_points(rnn, valid_predictions, batch_size=200): # Should be the same as the run_bayesRA code
	''' Find, analyze, and visualize the fixed points of the trained RNN.

	Args:
		model: FlipFlop object.

			Trained RNN model, as returned by train_FlipFlop().

		valid_predictions: dict.

			Model predictions on validation trials, as returned by
			train_FlipFlop().

	Returns:
		None.
	'''

	#NOISE_SCALE = 0.5 # Standard deviation of noise added to initial states
	NOISE_SCALE = 0.5 # Standard deviation of noise added to initial states
	N_INITS = batch_size # The number of initial states to provide

	'''Fixed point finder hyperparameters. See FixedPointFinder.py for detailed
	descriptions of available hyperparameters.'''
	fpf_hps = {
		'max_iters': 50000,
		'lr_init': 1., #originally set to 1
		'outlier_distance_scale': 10.0,
		'verbose': True, 
		'super_verbose': True}

	# Setup the fixed point finder
	fpf = FixedPointFinder(rnn, **fpf_hps)

	'''Draw random, noise corrupted samples of those state trajectories
	to use as initial states for the fixed point optimizations.'''
	initial_states = fpf.sample_states(valid_predictions,
		n_inits=N_INITS,
		noise_scale=NOISE_SCALE)

	#fps_load = np.load('/Users/dysprague/fixed-point-finder/data/fly_HD_varied_0_AV.npy')

	#row_max = np.max(fps_load, axis=1)
	#mask = row_max < 5

	#thresh_fps = fps_load[mask,:]

	#rng = np.random.RandomState()

	#oversample_fps = np.tile(thresh_fps, (100,1))
	#oversample_fps = oversample_fps[:N_INITS,:]

	#initial_states = oversample_fps + NOISE_SCALE * rng.randn(*oversample_fps.shape)

	mu_init, kappa_init = get_mu_kappa(initial_states.shape[1], initial_states) # mu_unit = batch x 1

	# Study the system in the absence of input pulses (e.g., all inputs are 0)
	inputs = np.zeros([batch_size, 2])

	inputs[:,0] = np.ones(batch_size)*np.pi/2 # set constant HD input
	#inputs[:,1] = np.ones(batch_size)*0.001 # set constant AV input

	HD_samples = np.linspace(-np.pi,np.pi-(2*np.pi)/80,80)
	#HD_samples = np.linspace(-np.pi/2,np.pi/2-(2*np.pi/2)/80,80)
	#HD_samples = mu_init + np.random.uniform(-np.pi/4+np.pi, np.pi/4+np.pi, len(mu_init))
	AV_samples = np.linspace(-0.002,0.002, 10)

	inputs[:,0] = np.random.choice(HD_samples, size=batch_size, replace=True) # sample HD inputs randomly from uniform options around circle
	#inputs[:,0] = HD_samples
	inputs[:,1] = np.random.choice(AV_samples, size=batch_size, replace=True)

	# Run the fixed point finder
	unique_fps, all_fps = fpf.find_fixed_points(initial_states, inputs)

	# Visualize identified fixed points with overlaid RNN state trajectories
	# All visualized in the 3D PCA space fit the the example RNN states.
	fig = plot_fps(unique_fps, valid_predictions,
		plot_batch_idx=list(range(10)),
		plot_start_time=10,
		plot_2d=True)
	
	return unique_fps

def main():
	#TODO: don't change main for the most part. Instead of training network, set parameters from paper

	T=20
	dt=0.001
	N=100
	kappa_phi=1 
	kappa_v=1
	kappa_z=15

	# Step 1: Train an RNN to solve the N-bit memory task
	trajectories, true_HD, HD_obs, AV_obs  = generate_data(T=T, dt=dt,N=N, kappa_phi=kappa_phi, kappa_v=kappa_v, kappa_z=kappa_z, batch_size=10) #TODO: get valid_predictions by running through network

	gen_inputs = torch.from_numpy(np.stack((HD_obs, AV_obs), axis=2))

	batch = 800
	
	rnn = drosophRNN(dt=dt, kappa_phi =kappa_phi, kappa_y=kappa_v) 

	initial_r_HD = torch.from_numpy(trajectories[:,0,:])

	initial_r_AVplus = rnn.w_AVplus_HD * torch.einsum('ij,bj->bi', rnn.W_AVplus_HD, initial_r_HD)
	initial_r_AVminus = rnn.w_AVplus_HD * torch.einsum('ij,bj->bi', rnn.W_AVminus_HD, initial_r_HD)

	initial_r_AVplus = torch.zeros(initial_r_AVplus.shape)
	initial_r_AVminus = torch.zeros(initial_r_AVminus.shape)

	initial_r_Del7 = 2 * rnn.w0 * rnn.kappa_0/(np.pi*(1-2*rnn.m0)) + rnn.w1 * rnn.kappa_0/(2*(1-rnn.m1))*torch.cos(rnn.phi_0_r_Del7-rnn.phi_0)
	initial_r_Del7 = initial_r_Del7.unsqueeze(0).repeat(initial_r_HD.shape[0],1)

	initial_state = torch.cat((initial_r_HD, initial_r_AVplus, initial_r_AVminus, initial_r_Del7), axis=1)

	initial_state = initial_state.unsqueeze(0)
	
	valid_predictions, final_state = rnn(gen_inputs, r0 = initial_state) # input with shape batch, time, num inputs; r0 with shape 1, batch, num neurons

	print(valid_predictions.shape)

	valid_predictions = valid_predictions[:,18000:,:]

	# STEP 2: Find, analyze, and visualize the fixed points of the trained RNN
	unique_fps = find_fixed_points(rnn, valid_predictions, batch_size=batch)

	fps = unique_fps.xstar

	fig = plt.figure()

	for fp in range(fps.shape[0]):
		plt.plot(fps[fp,:])

	plt.title('Fixed points')

	plt.show()

	print('Entering debug mode to allow interaction with objects and figures.')
	print('You should see a figure with:')
	print('\tMany blue lines approximately outlining a cube')
	print('\tStable fixed points (black dots) at corners of the cube')
	print('\tUnstable fixed points (red lines or crosses) '
		'on edges, surfaces and center of the cube')
	print('Enter q to quit.\n')
	pdb.set_trace()

if __name__ == '__main__':
	main()