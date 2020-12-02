import math
import random
import time
import os
import sys
from datetime import datetime

import collections
from collections import namedtuple
from collections import deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import numpy as np
import mcmc
from pymss import EnvManager
from IPython import embed
from Model import *
from RunningMeanStd import *
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

class PPO(object):
	def __init__(self,meta_file):
		np.random.seed(seed = int(time.time()))
		self.num_slaves = 900
		self.env = EnvManager(meta_file, self.num_slaves)
		self.use_muscle = self.env.UseMuscle()
		self.use_device = self.env.UseDevice()

		# ========== Character setting ========== #
		self.num_state = self.env.GetNumState()
		self.num_state_char = self.env.GetNumState_Char()
		self.num_state_device = 0
		if self.use_device:
			self.num_state_device = self.env.GetNumState_Device()
		self.num_action = self.env.GetNumAction()
		self.rms = RunningMeanStd(shape=(self.num_state))

		self.num_epochs = 10
		self.num_tuple_so_far = 0
		self.num_episode = 0
		self.num_tuple = 0

		print("char state : ", self.num_state)
		print("char action : ", self.num_action)
		self.model = SimulationNN(self.num_state,self.num_action)
		if use_cuda:
			self.model.cuda()
		# ========== Muscle setting ========= #
		if self.use_muscle:
			self.num_muscles = self.env.GetNumMuscles()
			self.num_action_muscle = self.env.GetNumAction()
			self.muscle_model = MuscleNN(self.env.GetNumTotalMuscleRelatedDofs(), self.num_action_muscle,self.num_muscles)
			if use_cuda:
				self.muscle_model.cuda()

		# ========== Common setting ========== #
		self.num_simulation_Hz = self.env.GetSimulationHz()
		self.num_control_Hz = self.env.GetControlHz()
		self.num_simulation_per_control = self.num_simulation_Hz // self.num_control_Hz

		self.env.Resets(True)

	def SaveModel(self):
		self.model.save('../nn/current.pt')
		self.rms.save('current')

		if self.max_return_epoch == self.num_evaluation:
			self.model.save('../nn/max.pt')
			self.rms.save('max')
		if self.num_evaluation%100 == 0:
			self.model.save('../nn/'+str(self.num_evaluation//100)+'.pt')
			self.rms.save(str(self.num_evaluation//100))

	def SaveModel_Muscle(self):
		self.muscle_model.save('../nn/current_muscle.pt')

		if self.max_return_epoch == self.num_evaluation:
			self.muscle_model.save('../nn/max_muscle.pt')
		if self.num_evaluation%self.save_interval == 0:
			self.muscle_model.save('../nn/'+str(self.num_evaluation//self.save_interval)+'_muscle.pt')

	def SaveModel_Marginal(self):
		self.marginal_model.save('../nn/current_marginal.pt')

		if self.max_return_epoch == self.num_evaluation:
			self.marginal_model.save('../nn/max_marginal.pt')
		if self.num_evaluation%self.save_interval == 0:
			self.marginal_model.save('../nn/'+str(self.num_evaluation//self.save_interval)+'_marginal.pt')

	def LoadModel(self, path):
		self.model.load('../nn/'+path+'.pt')
		self.rms.load(path)

	def LoadModel_Muscle(self,path):
		self.muscle_model.load('../nn/'+path+'_muscle.pt')

	def LoadModel_Muscle(self,path):
		self.marginal_model.load('../nn/'+path+'_marginal.pt')

	def Measure(self):
		self.counter = np.array([0 for i in range(900)])
		self.velocities = np.array([0.0 for i in range(900)])

		min_v = self.env.GetMinV()
		max_v = self.env.GetMaxV()

		states = [None]*self.num_slaves
		states = self.env.GetStates()

		delta1 = (max_v[0] - min_v[0])/30.0
		delta2 = (max_v[2] - min_v[2])/30.0

		for i in range(0, 30):
			for j in range(0, 30):
				param_state = np.float32(np.array([min_v[0]+delta1*i, 0.4, min_v[2]+delta2*j, 0.3]))
				self.env.SetParamState(i*30+j, param_state)
		# for i in range(self.num_slaves):
		# 	param_state = np.array([min_v[0]+0.1*(i/11), min_v[2]+0.08*(i%11)])
		# 	self.env.SetParamState(i, param_state)

		for i in range(300):
			a_dist,v = self.model(Tensor(states))
			actions = a_dist.sample().cpu().detach().numpy()
			logprobs = a_dist.log_prob(Tensor(actions)).cpu().detach().numpy().reshape(-1)
			values = v.cpu().detach().numpy().reshape(-1)
			self.env.SetActions(actions)
			if self.use_muscle:
				for i in range(self.num_simulation_per_control):
					mt = Tensor(self.env.GetMuscleTorques())
					self.env.SetDesiredTorques()
					dt = Tensor(self.env.GetDesiredTorques())
					activations = self.muscle_model(mt,dt).cpu().detach().numpy()
					self.env.SetActivationLevels(activations)
					self.env.Steps(1, self.use_device)
			else:
				self.env.StepsAtOnce(self.use_device)

			for j in range(self.num_slaves):
				nan_occur = False
				terminated_state = True

				if np.any(np.isnan(states[j])) or np.any(np.isnan(actions[j])) or np.any(np.isnan(values[j])) or np.any(np.isnan(logprobs[j])):
					nan_occur = True
				elif self.env.IsEndOfEpisode(j) is False:
					terminated_state = False
					self.velocities[j] += self.env.GetVelocity(j)*3.6
					self.counter[j] += 1

				if terminated_state:
					self.env.Reset(True,j)

			states = self.env.GetStates()
			states = self.rms.apply(states)

		for i in range(0, 900):
			if self.counter[i] == 0:
				pass
			else:
				self.velocities[i] /= float(self.counter[i])

		return self.velocities


import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import argparse
import os
if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-d','--meta',help='meta file')
	parser.add_argument('-m','--model',help='model path')
	parser.add_argument('-u','--muscle',help='muscle path')

	args =parser.parse_args()
	if args.meta is None:
		print('Provide meta file')
		exit()

	ppo = PPO(args.meta)
	nn_dir = '../nn'
	if not os.path.exists(nn_dir):
		os.makedirs(nn_dir)

	if args.model is not None:
		ppo.LoadModel(args.model)
	else:
		ppo.SaveModel()

	if args.muscle is not None:
		if ppo.use_muscle is False:
			print("Dont put : -u command")
			sys.exit()
		else:
			ppo.LoadModel_Muscle(args.muscle)
	else:
		if ppo.use_muscle:
			ppo.SaveModel_Muscle()

	print('num states: {}, num actions: {}'.format(ppo.env.GetNumState(),ppo.env.GetNumAction()))

	velocities = ppo.Measure()

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.zaxis.set_major_locator(LinearLocator(5))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

	X = []
	X.append(np.arange(0.5, 1.5, 1.0/30.0))
	X.append(np.arange(0.2, 1.0, 0.8/30.0))
	X[0], X[1] = np.meshgrid(X[0], X[1])

	V = np.zeros((30, 30))
	for i in range(30):
		for j in range(30):
			V[i][j] = velocities[i*30 + j]
	surf = ax.plot_surface(X[0], X[1], V, linewidth = 0, antialiased = False)

	plt.show()
