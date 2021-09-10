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

PARAM_DIV_NUM = 25
DIR_NAME = "../analysis/"
class PPO(object):
	def __init__(self,meta_file):
		np.random.seed(seed = int(time.time()))
		self.num_slaves = 1
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
		self.model = SimulationNN(self.num_state, self.num_action)
		if use_cuda:
			self.model.cuda()

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
			params_num = self.num_paramstate
			params_change_idx = []
			params_change_dim = 0
			for i in range(params_num):
				if min_v[i] != max_v[i]:
					params_change_idx.append(i)
					params_change_dim += 1

			self.params_norm = []
			self.params_real = []
			self.params_num = params_num
			self.params_change_dim = params_change_dim
			self.params_div_num = PARAM_DIV_NUM
			
			mul_v = (max_v-min_v)/(self.params_div_num)
			param_mul = 2.0/(self.params_div_num)
			if self.params_change_dim == 1:
				for i in range(self.params_div_num+1):
					params_norm = []
					params_real = []
					for j in range(params_num):
						if j == params_change_idx[0]:
							params_norm.append(-1 + i*param_mul)
							params_real.append(min_v[j] + i*mul_v[j])
						else:
							params_norm.append(0)
							params_real.append(min_v[j])
					self.params_norm.append(params_norm)
					self.params_real.append(params_real)

			if self.params_change_dim == 2:
				for i in range(self.param_div_num+1):
					for j in range(self.param_div_num+1):
						params_norm = []
						params_real = []
						for k in range(params_num):
							if k == params_change_idx[0]:
								params_norm.append(-1 + i*param_mul)
								params_real.append(min_v[k] + i*mul_v[k])
							elif k == params_change_idx[1]:
								params_norm.append(-1 + j*param_mul)
								params_real.append(min_v[k] + j*mul_v[k])
							else:
								params_norm.append(0)
								params_real.append(min_v[k])
						self.params_norm.append(params_norm)
						self.params_real.append(params_real)

			# print("size : ", len(self.params_real))
			# print("params : ", self.params)
	
		# ============Analysis==============
		if self.use_adaptive_sampling:
			add_env_num = len(self.params_norm) - self.num_slaves
			self.env.AddEnvironments(add_env_num)
			self.num_slaves += add_env_num
		
		self.analysisDatasize = 30
		self.analysisData = [None]*self.num_slaves
		for j in range(self.num_slaves):
			self.analysisData[j] = DataBuffer()

	def LoadModel(self, path):
		self.model.load('../nn/'+path+'.pt')
		self.rms.load(path)

	def LoadModel_Muscle(self,path):
		self.muscle_model.load('../nn/'+path+'_muscle.pt')

	def GaitGenerate(self):
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
				self.env.SetParamState(j, self.params_norm[j])
		
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
					if (reward[j] > 0.3 and self.env.isAnalysisPeriod(j)):
						velocity[j] = self.env.GetVelocity(j)
						stride[j] = self.env.GetStride(j)
						cadence[j] = self.env.GetCadence(j)
						torqueEnergy[j] = self.env.GetTorqueEnergy(j)
						stanceRatio[j] = self.env.GetStanceRatioRight(j)
						gaitTime[j] = self.env.GetGaitTimeRight(j)

						self.analysisData[j].Push(reward[j], velocity[j], stride[j], cadence[j], torqueEnergy[j], stanceRatio[j], gaitTime[j])	
						local_step[j] += 1

				if self.env.isEndAnalysisPeriod(j) or terminated_state or nan_occur:
					self.env.Reset(True,j)
					if self.use_adaptive_sampling:
						self.env.SetParamState(j, self.params_norm[j])

			isDone = True
			for j in range(self.num_slaves):
				if local_step[j] < self.analysisDatasize:
					isDone = False
			
			if isDone:
				break

			states = self.env.GetStates()
			states = self.rms.apply(states)

	def GaitAnalysis(self):
		fileName = DIR_NAME + "analysis.txt"
		os.makedirs(os.path.dirname(fileName), exist_ok=True)
		
		f = open(fileName, 'w')
		f.write("idx ")
		if self.use_adaptive_sampling:
			for i in range(self.params_num):
				f.write("p"+str(i)+" ")
		f.write("Rew Vel Str Cdc tE StrRtio GaitT StncT" + "\n")
		
		div = 0
		if self.use_adaptive_sampling:
			div = len(self.params_norm)			
		else:
			div = 1

		print("div : ", div)
		for i in range(div):
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

			f.write("%d " % (i))
			if self.use_adaptive_sampling:
				for j in range(self.params_num):
					f.write("%f " % (self.params_real[i][j]))
			
			f.write("%f %f %f %f %f %f %f %f\n" % (r, v, s, c, te, sr, gt, st))
			# print(i, " reward : ", np.mesan(np.array(reward)))
			# print(i, " velocity : ", np.mean(np.array(velocity)))
			# print(i, " strides : ", np.mean(np.array(stride)))
			# print(i, " cadence : ", np.mean(np.array(cadence)))
			# print(i, " energy : ", np.mean(np.array(torqueEnergy)))
			# print("\n")

		f.close()

		div1 = 6
		div2 = 26
		if self.use_adaptive_sampling:
			if self.params_change_dim == 2:
				r2 = np.empty((div1, div2))
				v2 = np.empty((div1, div2))
				s2 = np.empty((div1, div2))
				c2 = np.empty((div1, div2))
				te2 = np.empty((div1, div2))
				sr2 = np.empty((div1, div2))
				gt2 = np.empty((div1, div2))
				st2 = np.empty((div1, div2))	

				
				fileName = DIR_NAME + "analysis2.txt"
				os.makedirs(os.path.dirname(fileName), exist_ok=True)
		
				f = open(fileName, 'w')
				for i in range(div2):
					for j in range(div1):
						idx = i * div2 + j * div1 - 1
						data = self.analysisData[idx].GetData()
						
						reward, velocity, stride, cadence, torqueEnergy, stanceRatio, gaitTime = zip(*data)
						r2[j][i] = np.mean(np.array(reward))
						v2[j][i] = np.mean(np.array(velocity))
						s2[j][i] = np.mean(np.array(stride))
						c2[j][i] = np.mean(np.array(cadence))
						te2[j][i] = np.mean(np.array(torqueEnergy))
						sr2[j][i] = np.mean(np.array(stanceRatio))
						gt2[j][i] = np.mean(np.array(gaitTime))
						st2[j][i] = sr2[j][i] * gt2[j][i] #stance time
				
				for i in range(div2):
					a = (1.0/25.0)*i
					f.write("%f " % a)
				f.write("\n")

				f.write("Rew" + "\n")
				for i in range(div1):
					f.write("%f " % (0.1*i))
					for j in range(div2):
						f.write("%f " % r2[i][j])
					f.write("\n")

				f.write("Vel" + "\n")
				for i in range(div1):
					f.write("%f " % (0.1*i))
					for j in range(div2):
						f.write("%f " % v2[i][j])
					f.write("\n")

				f.write("Str" + "\n")
				for i in range(div1):
					f.write("%f " % (0.1*i))
					for j in range(div2):
						f.write("%f " % s2[i][j])
					f.write("\n")

				f.write("Cdc" + "\n")
				for i in range(div1):
					f.write("%f " % (0.1*i))
					for j in range(div2):
						f.write("%f " % c2[i][j])
					f.write("\n")

				f.write("tE" + "\n")
				for i in range(div1):
					f.write("%f " % (0.1*i))
					for j in range(div2):
						f.write("%f " % te2[i][j])
					f.write("\n")

				f.write("strRtio" + "\n")
				for i in range(div1):
					f.write("%f " % (0.1*i))
					for j in range(div2):
						f.write("%f " % sr2[i][j])
					f.write("\n")

				f.write("gaitT" + "\n")
				for i in range(div1):
					f.write("%f " % (0.1*i))
					for j in range(div2):
						f.write("%f " % gt2[i][j])
					f.write("\n")

				f.write("stncT" + "\n")
				for i in range(div1):
					f.write("%f " % (0.1*i))
					for j in range(div2):
						f.write("%f " % st2[i][j])
					f.write("\n")
				
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
