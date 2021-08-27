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
from pywad import EnvManager
from IPython import embed
from Model import *
from RunningMeanStd import *
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import argparse
import os

MyData = namedtuple('Data',('reward','velocity','stride', 'cadence', 'torqueEnergy'))
class DataBuffer(object):
	def __init__(self):
		self.data = []
	def Push(self, *args):
		self.data.append(MyData(*args))
	def Pop(self):
		self.data.pop()
	def GetData(self):
		return self.data
	def Size(self):
		return len(self.data)

PARAM_DIV_NUM = 7
PARAM_DIM = 1
class PPO(object):
	def __init__(self,meta_file):
		np.random.seed(seed = int(time.time()))
		self.num_slaves = int(math.pow(PARAM_DIV_NUM, PARAM_DIM))
		self.env = EnvManager(meta_file, self.num_slaves)
		self.use_muscle = self.env.UseMuscle()
		self.use_device = self.env.UseDevice()
		self.use_adaptive_motion = self.env.UseAdaptiveMotion()
		
		# ========== Character setting ========== #
		self.num_state = self.env.GetNumState()
		self.num_state_char = self.env.GetNumState_Char()
		self.num_state_device = 0
		if self.use_device:
			self.num_state_device = self.env.GetNumState_Device()
		self.num_action = self.env.GetNumAction()
		self.num_active_dof = self.env.GetNumActiveDof()
		self.rms = RunningMeanStd(shape=(self.num_state))

		print("char state : ", self.num_state)
		print("char action : ", self.num_action)
		self.model = SimulationNN(self.num_state,self.num_action)
		if use_cuda:
			self.model.cuda()

		self.analysisDatasize = 64*self.num_slaves
		self.analysisData = [None]*self.num_slaves
		for j in range(self.num_slaves):
			self.analysisData[j] = DataBuffer()

		# ========== Muscle setting ========= #
		if self.use_muscle:
			self.num_muscles = self.env.GetNumMuscles()
			self.num_action_muscle = self.env.GetNumActiveDof()
			self.muscle_model = MuscleNN(self.env.GetNumTotalMuscleRelatedDofs(), self.num_action_muscle, self.num_muscles)
			
			if use_cuda:
				self.muscle_model.cuda()

		# ===== Adaptive Sampling setting ==== #
		self.use_adaptive_sampling = self.env.UseAdaptiveSampling()
		if self.use_adaptive_sampling:
			self.num_paramstate = self.env.GetNumParamState()
			self.num_paramstate_char = self.env.GetNumParamState_Char()
			if self.use_device:
				self.num_paramstate_device = self.env.GetNumParamState_Device()
			
			min_v = self.env.GetMinV()
			max_v = self.env.GetMaxV()
			param_num = len(min_v)
			param_idx = []
			for i in range(param_num):
				if min_v[i] != max_v[i]:
					param_idx.append(i)

			self.params = []
			self.param_div_num = PARAM_DIV_NUM

			param_mul = 2.0/self.param_div_num
			param_change_num = len(param_idx)
			if param_change_num == 1:
				for i in range(self.param_div_num + 1):
					params = []
					for j in range(self.num_paramstate):
						if j == param_idx[0]:
							params.append(-1 + i*param_mul)
						else:
							params.append(0)
					self.params.append(params)

			if param_change_num == 2:
				for i in range(self.param_div_num + 1):
					for j in range(self.param_div_num + 1):
						params = []
						for k in range(self.num_paramstate):
							if k == param_idx[0]:
								params.append(-1 + i*param_mul)
							elif k == param_idx[1]:
								params.append(-1 + j*param_mul)
							else:
								params.append(0)
						self.params.append(params)

			# print("params : ", self.params)
	
	def LoadModel(self, path):
		self.model.load('../nn/'+path+'.pt')
		self.rms.load(path)

	def LoadModel_Muscle(self,path):
		self.muscle_model.load('../nn/'+path+'_muscle.pt')

	def GaitGenerate(self):
		self.totalData = []
		velocity = [None]*self.num_slaves
		stride = [None]*self.num_slaves
		cadence = [None]*self.num_slaves
		torqueEnergy = [None]*self.num_slaves
		
		states = self.env.GetStates()

		if self.use_adaptive_sampling:
			for j in range(self.num_slaves):
				self.env.SetParamState(j, self.params[j])
		
		counter = 0
		local_step = 0
		while True:
			counter += 1
			if counter%10 == 0:
				print('SIM : {}'.format(local_step),end='\r')

			a_dist,v = self.model(Tensor(states))
			actions = a_dist.sample().cpu().detach().numpy()
			logprobs = a_dist.log_prob(Tensor(actions)).cpu().detach().numpy().reshape(-1)
			values = v.cpu().detach().numpy().reshape(-1)
			self.env.SetActions(actions)

			if self.use_muscle:
				mt = Tensor(self.env.GetMuscleTorques())
				for i in range(self.num_simulation_per_control):
					self.env.SetDesiredTorques()
					dt = Tensor(self.env.GetDesiredTorques())
					activations = self.muscle_model(mt,dt).cpu().detach().numpy()
					self.env.SetActivationLevels(activations)
					self.env.Steps(self.inference_per_sim, self.use_device)
			else:
				self.env.StepsAtOnce(self.use_device)

			for j in range(self.num_slaves):
				nan_occur = False
				terminated_state = True
				if np.any(np.isnan(states[j])) or np.any(np.isnan(actions[j])) or np.any(np.isnan(values[j])) or np.any(np.isnan(logprobs[j])):
					nan_occur = True
				elif self.env.IsEndOfEpisode(j) is False:
					terminated_state = False
					rewards[j] = self.env.GetReward(j)
					reward = rewards[j][0]
					if (reward > 0.5 && self.env.isAnalysisPeriod(j):
						velocity[j] = self.env.GetVelocity(j)
						stride[j] = self.env.GetStride(j)
						cadence[j] = self.env.GetCadence(j)
						torqueEnergy[j] = self.env.GetTorqueEnergy(j)

						self.analysisData[j].Push(reward, velocity[j], stride[j], cadence[j], torqueEnergy[j])	
					local_step += 1

				if self.env.isEndAnalysisPeriod():
					self.totalData.append(self.analysisData[j])
					self.env.Reset(True,j)
					if self.use_adaptive_sampling:
						self.env.SetParamState(j, self.params[j])
					
			if local_step >= self.analysisDatasize:
				break

			states = self.env.GetStates()
			states = self.rms.apply(states)

	def GaitAnalysis(self):
		for epi in self.totalData:
			data = epi.GetData()
			size = len(data)
			if size == 0:
				continue
			
			reward, velocity, stride, cadence, torqueEnergy = zip(*data)
						

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
		print("provie nn.pt")

	if args.muscle is not None:
		if ppo.use_muscle is False:
			print("Dont put : -u command")
			sys.exit()
		else:
			ppo.LoadModel_Muscle(args.muscle)
	else:
		if ppo.use_muscle:
			print("provide muscle_nn.pt")

	print('num states: {}, num actions: {}'.format(ppo.env.GetNumState(),ppo.env.GetNumAction()))
	# ppo.GaitGenerate()
	# ppo.GaitAnalysis()
	# ppo.Plot()

