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

MyData = namedtuple('Data',('reward','velocity','stride', 'cadence', 'torqueEnergy', 'stanceRatio', 'gaitTime'))
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

PARAM_DIV_NUM = 1
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

		self.analysisDatasize = 300*self.num_slaves
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
			
			min_v = np.array(self.env.GetMinV())
			max_v = np.array(self.env.GetMaxV())
			param_num = len(min_v)
			param_idx = []
			for i in range(param_num):
				if min_v[i] != max_v[i]:
					param_idx.append(i)

			self.params = []
			self.params_real = []
			self.param_div_num = PARAM_DIV_NUM
			
			mul_v = (max_v-min_v)/(self.param_div_num-1)
			for i in range(self.param_div_num):
				params = min_v + i*mul_v
				params_ = params.tolist()
				self.params_real.append(params_)
			
			param_mul = 2.0/(self.param_div_num-1)
			param_change_num = len(param_idx)
			if param_change_num == 1:
				for i in range(self.param_div_num):
					params = []
					for j in range(self.num_paramstate):
						if j == param_idx[0]:
							params.append(-1 + i*param_mul)
						else:
							params.append(0)
					self.params.append(params)

			if param_change_num == 2:
				for i in range(self.param_div_num):
					for j in range(self.param_div_num):
						params = []
						params_real = []
						for k in range(self.num_paramstate):
							if k == param_idx[0]:
								params.append(-1 + i*param_mul)
								params_real.append(min_v[k] + i*mul_v[k])
							elif k == param_idx[1]:
								params.append(-1 + j*param_mul)
								params_real.append(min_v[k] + j*mul_v[k])
							else:
								params.append(0)
								params.real.append(min_v[k])
						self.params.append(params)
						self.params_real.append(params_real)

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
		reward = [None]*self.num_slaves
		stanceRatio = [None]*self.num_slaves
		gaitTime = [None]*self.num_slaves
		
		states = self.env.GetStates()

		if self.use_adaptive_sampling:
			for j in range(self.num_slaves):
				self.env.SetParamState(j, self.params[j])
		
		counter = 0
		local_step = [0]*self.num_slaves
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
					# rewards[j] = self.env.GetReward(j)
					# reward = rewards[j][0]
					reward[j] = self.env.GetReward(j)[0] 					
					if (reward[j] > 0.5 and self.env.isAnalysisPeriod(j)):
						velocity[j] = self.env.GetVelocity(j)
						stride[j] = self.env.GetStride(j)
						cadence[j] = self.env.GetCadence(j)
						torqueEnergy[j] = self.env.GetTorqueEnergy(j)
						stanceRatio[j] = self.env.GetStanceRatioRight(j)
						gaitTime[j] = self.env.GetGaitTimeRight(j)

						self.analysisData[j].Push(reward[j], velocity[j], stride[j], cadence[j], torqueEnergy[j], stanceRatio[j], gaitTime[j])	
						local_step[j] += 1

				if self.env.isEndAnalysisPeriod(j) or terminated_state or nan_occur:
					# self.totalData.append(self.analysisData[j])
					self.env.Reset(True,j)
					if self.use_adaptive_sampling:
						self.env.SetParamState(j, self.params[j])

			isDone = True
			for j in range(self.num_slaves):
				if local_step[j] < self.analysisDatasize:
					isDone = False
			
			if isDone:
				break

			states = self.env.GetStates()
			states = self.rms.apply(states)

	def GaitAnalysis(self):

		f = open("analysis.txt", 'w')
		if self.use_adaptive_sampling:
			f.write("idx p0 p1 p2 p3 p4 r v s c te sr gt" + "\n")
		else:
			f.write("idx r v s c te sr gt" + "\n")
		for i in range(self.num_slaves):
			data = self.analysisData[i].GetData()
			size = len(data)

			reward, velocity, stride, cadence, torqueEnergy, stanceRatio, gaitTime = zip(*data)
			r = np.mean(np.array(reward))
			v = np.mean(np.array(velocity))
			s = np.mean(np.array(stride))
			c = np.mean(np.array(cadence))
			te = np.mean(np.array(torqueEnergy))
			sr = np.mean(np.array(stanceRatio))
			gt = np.mean(np.array(gaitTime))
			st = sr * gt #stance time

			if self.use_adaptive_sampling:
				f.write("%d %f %f %f %f %f " % (i, self.params_real[i][0], self.params_real[i][1], self.params_real[i][2], self.params_real[i][3], self.params_real[i][4]))
			else:
				f.write("%d " % (i))
			f.write("%f %f %f %f %f %f %f %f\n" % (r, v, s, c, te, sr, gt, st))
			# print(i, " reward : ", np.mesan(np.array(reward)))
			# print(i, " velocity : ", np.mean(np.array(velocity)))
			# print(i, " strides : ", np.mean(np.array(stride)))
			# print(i, " cadence : ", np.mean(np.array(cadence)))
			# print(i, " energy : ", np.mean(np.array(torqueEnergy)))
			# print("\n")

		f.close()

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
	ppo.GaitGenerate()
	ppo.GaitAnalysis()
	# ppo.Plot()
