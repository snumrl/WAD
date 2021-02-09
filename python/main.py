import os
import sys
import math
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import collections
from collections import namedtuple
from collections import deque
from itertools import count
from datetime import datetime

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

Episode = namedtuple('Episode',('s','a','r', 'value', 'logprob'))
class EpisodeBuffer(object):
	def __init__(self):
		self.data = []

	def Push(self, *args):
		self.data.append(Episode(*args))
	def Pop(self):
		self.data.pop()
	def GetData(self):
		return self.data
	def Size(self):
		return len(self.data)

MuscleTransition = namedtuple('MuscleTransition',('JtA','tau_des','L','b'))
class MuscleBuffer(object):
	def __init__(self, buff_size = 10000):
		super(MuscleBuffer, self).__init__()
		self.buffer = deque(maxlen=buff_size)

	def Push(self,*args):
		self.buffer.append(MuscleTransition(*args))

	def Clear(self):
		self.buffer.clear()

Transition = namedtuple('Transition',('s','a', 'logprob', 'TD', 'GAE'))
class ReplayBuffer(object):
	def __init__(self, buff_size = 10000):
		super(ReplayBuffer, self).__init__()
		self.buffer = deque(maxlen=buff_size)

	def Push(self,*args):
		self.buffer.append(Transition(*args))

	def Clear(self):
		self.buffer.clear()

# Add Adaptive Sampling
MarginalTransition = namedtuple('MarginalTransition', ('sb', 'v'))
class MarginalBuffer(object):
    def __init__(self, buff_size = 10000):
        super(MarginalBuffer, self).__init__()
        self.buffer = deque(maxlen=buff_size)

    def Push(self, *_args):
        self.buffer.append(MarginalTransition(*_args))

    def Clear(self):
        self.buffer.clear()

class PPO(object):
	def __init__(self,meta_file):
		np.random.seed(seed = int(time.time()))
		self.num_slaves = 16
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

		self.buffer_size = 1024*8
		self.batch_size = 128*8
		self.replay_buffer = ReplayBuffer(30000)

		self.gamma = 0.99
		self.lb = 0.99

		self.default_clip_ratio = 0.2
		self.default_learning_rate = 1.0*1E-4
		self.clip_ratio = self.default_clip_ratio
		self.learning_rate = self.default_learning_rate

		self.optimizer = optim.Adam(self.model.parameters(),lr=self.learning_rate)

		self.w_entropy = -0.001
		self.loss_actor = 0.0
		self.loss_critic = 0.0
		self.sum_return = 0.0
		self.max_return = -1.0
		self.max_return_epoch = 1

		self.episodes = [None]*self.num_slaves
		for j in range(self.num_slaves):
			self.episodes[j] = EpisodeBuffer()

		# ========== Muscle setting ========= #
		if self.use_muscle:
			self.num_muscles = self.env.GetNumMuscles()
			self.num_action_muscle = self.env.GetNumAction()
			self.muscle_model = MuscleNN(self.env.GetNumTotalMuscleRelatedDofs(), self.num_action_muscle,self.num_muscles)
			self.muscle_buffer = MuscleBuffer(30000)
			# self.rms_muscle = RunningMeanStd(shape=(self.env.GetNumTotalMuscleRelatedDofs()))

			self.loss_muscle = 0.0
			self.muscle_batch_size = 128
			self.default_learning_rate_muscle = 1E-4
			self.learning_rate_muscle = self.default_learning_rate_muscle
			self.optimizer_muscle = optim.Adam(self.muscle_model.parameters(),lr=self.learning_rate_muscle)
			self.num_epochs_muscle = 3
			if use_cuda:
				self.muscle_model.cuda()

		# ===== Adaptive Sampling setting ==== #

		self.use_adaptive_sampling = self.env.UseAdaptiveSampling()
		if self.use_adaptive_sampling:
			self.num_paramstate = self.env.GetNumParamState()
			self.num_paramstate_char = self.env.GetNumParamState_Char()
			if self.use_device:
				self.num_paramstate_device = self.env.GetNumParamState_Device()
			self.marginal_buffer = MarginalBuffer(10000)
			self.marginal_model = MarginalNN(self.num_paramstate)
			self.marginal_value_avg = 1.
			self.marginal_learning_rate = 5e-5
			if use_cuda:
				self.marginal_model.cuda()
			self.marginal_optimizer = optim.Adam(self.marginal_model.parameters(), lr=self.marginal_learning_rate)
			self.marginal_loss = 0.0

			self.InitialParamStates = []
			self.InitialParamStates_num = 1000

			self.marginal_k = 5

		# ========== Common setting ========== #
		self.num_simulation_Hz = self.env.GetSimulationHz()
		self.num_control_Hz = self.env.GetControlHz()
		self.inference_per_sim = 2
		self.num_simulation_per_control = (self.num_simulation_Hz // self.num_control_Hz) // self.inference_per_sim

		self.rewards = []
		self.max_iteration = 10000
		self.num_evaluation = 0
		self.save_interval = 100

		self.tic = time.time()
		self.env.Resets(True)

	def SaveModel(self):
		self.model.save('../nn/current.pt')
		self.rms.save('current')

		if self.max_return_epoch == self.num_evaluation:
			self.model.save('../nn/max.pt')
			self.rms.save('max')
		if self.num_evaluation%self.save_interval == 0:
			self.model.save('../nn/'+str(self.num_evaluation//self.save_interval)+'.pt')
			self.rms.save(str(self.num_evaluation//self.save_interval))

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

	def LoadModel_Marginal(self,path):
		self.marginal_model.load('../nn/'+path+'_marginal.pt')

	def Train(self):
		if self.use_adaptive_sampling:
			self.GenerateInitialStates()
		self.GenerateTransitions()
		self.OptimizeModel()

	def OptimizeModel(self):
		self.ComputeTDandGAE()
		self.OptimizeSimulationNN()
		if self.use_muscle:
			self.OptimizeMuscleNN()
		if self.use_adaptive_sampling:
			self.OptimizeMarginalNN()

	def GenerateInitialStates(self):
		min_v = self.env.GetMinV()
		max_v = self.env.GetMaxV()

		# target distribution
		def target_dist(x):
			marginal_value = self.marginal_model(Tensor(x)).cpu().detach().numpy().reshape(-1)
			p = math.exp(self.marginal_k * (1. - marginal_value/self.marginal_value_avg))
			return p

		# inverse_target distribution
		def inverse_target_dist(x):
			marginal_value = self.marginal_model(Tensor(x)).cpu().detach().numpy().reshape(-1)
			p = -math.exp(self.marginal_k * (1. - marginal_value/self.marginal_value_avg))
			return p

		# proposed distribution
		def proposed_dist(x, min_v, max_v):
			size = x.size
			value = []
			for i in range(size):
				if min_v[i] == max_v[i]:
					value.append(min_v[i])
				else:
					value.append(np.random.uniform(-1.0, 1.0))

			value = np.array(value)
			return value

		mcmc_sampler = mcmc.MetropolisHasting(self.num_paramstate, min_v, max_v, target_dist, proposed_dist)

		self.InitialParamStates.clear()
		self.InitialParamStates = mcmc_sampler.get_sample(self.InitialParamStates_num)

	def GenerateTransitions(self):
		self.total_episodes = []
		actions = [None]*self.num_slaves
		rewards = [None]*self.num_slaves
		states = [None]*self.num_slaves
		states_next = [None]*self.num_slaves
		terminated = [False]*self.num_slaves


		states = self.env.GetStates()

		if self.use_adaptive_sampling:
			for j in range(self.num_slaves):
				initial_state = np.float32(random.choice(self.InitialParamStates))
				self.env.SetParamState(j, initial_state)

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
					self.episodes[j].Push(states[j], actions[j], rewards[j], values[j], logprobs[j])
					local_step += 1

				if terminated_state or (nan_occur is True):
					if (nan_occur is True):
						self.episodes[j].Pop()
					else:
						self.total_episodes.append(self.episodes[j])
					self.episodes[j] = EpisodeBuffer()

					self.env.Reset(True,j)
					if self.use_adaptive_sampling:
						initial_state = np.float32(random.choice(self.InitialParamStates))
						self.env.SetParamState(j, initial_state)

			if local_step >= self.buffer_size:
				break

			states = self.env.GetStates()
			states = self.rms.apply(states)

	def ComputeTDandGAE(self):
		self.replay_buffer.Clear()
		if self.use_muscle:
			self.muscle_buffer.Clear()
		if self.use_adaptive_sampling:
			self.marginal_buffer.Clear()

		self.sum_return = 0.0
		for epi in self.total_episodes:
			data = epi.GetData()
			size = len(data)
			if size == 0:
				continue
			states, actions, rewards, values, logprobs = zip(*data)

			values = np.concatenate((values, np.zeros(1)), axis=0)
			advantages = np.zeros(size)
			ad_t = 0

			epi_return = 0.0
			for i in reversed(range(len(data))):
				epi_return += rewards[i]
				delta = rewards[i] + values[i+1] * self.gamma - values[i]
				ad_t = delta + self.gamma * self.lb * ad_t
				advantages[i] = ad_t
			self.sum_return += epi_return
			TD = values[:size] + advantages

			for i in range(size):
				self.replay_buffer.Push(states[i], actions[i], logprobs[i], TD[i], advantages[i])

			if self.use_adaptive_sampling:
				for i in range(size):
					cur_state = states[i][self.num_state_char-self.num_paramstate_char:self.num_state_char]
					if self.use_device:
						cur_state = np.append(cur_state, states[i][self.num_state-self.num_paramstate_device:self.num_state])

					self.marginal_buffer.Push(cur_state, values[i])

		self.num_episode = len(self.total_episodes)
		self.num_tuple = len(self.replay_buffer.buffer)
		print('SIM : {}'.format(self.num_tuple))
		self.num_tuple_so_far += self.num_tuple

		if self.use_muscle:
			muscle_tuples = self.env.GetMuscleTuples()
			for i in range(len(muscle_tuples)):
				self.muscle_buffer.Push(muscle_tuples[i][0],muscle_tuples[i][1],muscle_tuples[i][2],muscle_tuples[i][3])

	def OptimizeSimulationNN(self):
		all_transitions = np.array(self.replay_buffer.buffer, dtype=object)
		for j in range(self.num_epochs):
			np.random.shuffle(all_transitions)
			for i in range(len(all_transitions)//self.batch_size):
				transitions = all_transitions[i*self.batch_size:(i+1)*self.batch_size]
				batch = Transition(*zip(*transitions))

				stack_s = np.vstack(batch.s).astype(np.float32)
				stack_a = np.vstack(batch.a).astype(np.float32)
				stack_lp = np.vstack(batch.logprob).astype(np.float32)
				stack_td = np.vstack(batch.TD).astype(np.float32)
				stack_gae = np.vstack(batch.GAE).astype(np.float32)

				a_dist,v = self.model(Tensor(stack_s))

				loss_critic = ((v-Tensor(stack_td)).pow(2)).mean()

				'''Actor Loss'''
				ratio = torch.exp(a_dist.log_prob(Tensor(stack_a))-Tensor(stack_lp))
				stack_gae = (stack_gae-stack_gae.mean())/(stack_gae.std()+ 1E-5)
				stack_gae = Tensor(stack_gae)
				surrogate1 = ratio * stack_gae
				surrogate2 = torch.clamp(ratio,min =1.0-self.clip_ratio,max=1.0+self.clip_ratio) * stack_gae
				loss_actor = - torch.min(surrogate1,surrogate2).mean()

				'''Entropy Loss'''
				loss_entropy = - self.w_entropy * a_dist.entropy().mean()

				self.loss_actor = loss_actor.cpu().detach().numpy().tolist()
				self.loss_critic = loss_critic.cpu().detach().numpy().tolist()

				loss = loss_actor + loss_entropy + loss_critic

				self.optimizer.zero_grad()
				loss.backward(retain_graph=True)
				for param in self.model.parameters():
					if param.grad is not None:
						param.grad.data.clamp_(-0.5,0.5)
				self.optimizer.step()

			print('Optimizing sim nn : {}/{}'.format(j+1,self.num_epochs),end='\r')
		print('')

	def OptimizeMuscleNN(self):
		muscle_transitions = np.array(self.muscle_buffer.buffer, dtype=object)
		for j in range(self.num_epochs_muscle):
			np.random.shuffle(muscle_transitions)
			for i in range(len(muscle_transitions)//self.muscle_batch_size):
				tuples = muscle_transitions[i*self.muscle_batch_size:(i+1)*self.muscle_batch_size]
				batch = MuscleTransition(*zip(*tuples))

				stack_JtA = np.vstack(batch.JtA).astype(np.float32)
				stack_tau_des = np.vstack(batch.tau_des).astype(np.float32)
				stack_L = np.vstack(batch.L).astype(np.float32)
				stack_L = stack_L.reshape(self.muscle_batch_size, self.num_action, self.num_muscles)
				stack_b = np.vstack(batch.b).astype(np.float32)

				stack_JtA = Tensor(stack_JtA)
				stack_tau_des = Tensor(stack_tau_des)
				stack_L = Tensor(stack_L)
				stack_b = Tensor(stack_b)

				activation = self.muscle_model(stack_JtA,stack_tau_des)
				tau = torch.einsum('ijk,ik->ij',(stack_L,activation)) + stack_b

				loss_reg = (activation).pow(2).mean()
				loss_target = (((tau-stack_tau_des)/100.0).pow(2)).mean()
				loss = 0.05*loss_reg + loss_target

				self.optimizer_muscle.zero_grad()
				loss.backward(retain_graph=True)
				for param in self.muscle_model.parameters():
					if param.grad is not None:
						param.grad.data.clamp_(-0.5,0.5)
				self.optimizer_muscle.step()

			print('Optimizing muscle nn : {}/{}'.format(j+1,self.num_epochs_muscle),end='\r')
		self.loss_muscle = loss.cpu().detach().numpy().tolist()
		print('')

	def OptimizeMarginalNN(self):
		marginal_transitions = np.array(self.marginal_buffer.buffer, dtype = object)
		for j in range(self.num_epochs):
			np.random.shuffle(marginal_transitions)
			for i in range(len(marginal_transitions)//self.batch_size):
				transitions = marginal_transitions[i*self.batch_size:(i+1)*self.batch_size]
				batch = MarginalTransition(*zip(*transitions))

				stack_sb = np.vstack(batch.sb).astype(np.float32)
				stack_v = np.vstack(batch.v).astype(np.float32)

				# embed()
				v = self.marginal_model(Tensor(stack_sb))

				# Marginal Loss
				loss_marginal = ((v-Tensor(stack_v)).pow(2)).mean()
				self.marginal_loss = loss_marginal.cpu().detach().numpy().tolist()
				self.marginal_optimizer.zero_grad()
				loss_marginal.backward(retain_graph=True)

				for param in self.marginal_model.parameters():
					if param.grad is not None:
						param.grad.data.clamp_(-0.5, 0.5)
				self.marginal_optimizer.step()

				# Marginal value average
				avg_marginal = Tensor(stack_v).mean().cpu().detach().numpy().tolist()
				self.marginal_value_avg -= self.marginal_learning_rate * (self.marginal_value_avg - avg_marginal)

			print('Optimizing margin nn : {}/{}'.format(j+1, self.num_epochs), end='\r')
		print('')

	def Evaluate(self):
		self.num_evaluation = self.num_evaluation + 1
		h = int((time.time() - self.tic)//3600.0)
		m = int((time.time() - self.tic)//60.0)
		s = int((time.time() - self.tic))
		m = m - h*60
		s = int((time.time() - self.tic))
		s = s - h*3600 - m*60

		if self.num_episode == 0:
			self.num_episode = 1
		if self.num_tuple == 0:
			self.num_tuple = 1
		if self.max_return < self.sum_return/self.num_episode:
			self.max_return = self.sum_return/self.num_episode
			self.max_return_epoch = self.num_evaluation

		print('# {} === {}h:{}m:{}s ==='.format(self.num_evaluation,h,m,s))
		print('||Loss Actor               : {:.4f}'.format(self.loss_actor))
		print('||Loss Critic              : {:.4f}'.format(self.loss_critic))
		if self.use_muscle:
			print('||Loss Muscle              : {:.4f}'.format(self.loss_muscle))
		print('||Noise                    : {:.3f}'.format(self.model.log_std.exp().mean()))
		print('||Num Transition So far    : {}'.format(self.num_tuple_so_far))
		print('||Num Transition           : {}'.format(self.num_tuple))
		print('||Num Episode              : {}'.format(self.num_episode))
		print('||Avg Transition / episode : {}'.format(int(self.num_tuple/self.num_episode)))
		print('||Avg Return per episode   : {:.3f}'.format(self.sum_return/self.num_episode))
		print('||Avg Reward per transition: {:.3f}'.format(self.sum_return/self.num_tuple))
		print('||Avg Step per episode     : {:.1f}'.format(self.num_tuple/self.num_episode))
		print('||Max Avg Retun So far     : {:.3f} at #{}'.format(self.max_return, self.max_return_epoch))
		print('=============================================')

		self.SaveModel()
		if self.use_muscle:
			self.SaveModel_Muscle()
		if self.use_adaptive_sampling:
			self.SaveModel_Marginal()

		self.rewards.append(self.sum_return/self.num_episode)
		return np.array(self.rewards)

import matplotlib
import matplotlib.pyplot as plt
plt.ion()
def Plot(y,title,num_fig=1,ylim=True,save=False):
	temp_y = np.zeros(y.shape)
	if y.shape[0]>5:
		temp_y[0] = y[0]
		temp_y[1] = 0.5*(y[0] + y[1])
		temp_y[2] = 0.3333*(y[0] + y[1] + y[2])
		temp_y[3] = 0.25*(y[0] + y[1] + y[2] + y[3])
		for i in range(4,y.shape[0]):
			temp_y[i] = np.sum(y[i-4:i+1])*0.2

	plt.figure(num_fig)
	plt.clf()
	plt.title(title)
	plt.plot(y,'b')

	plt.plot(temp_y,'r')

	plt.show()
	if ylim:
		plt.ylim([0,1])
	if save:
		x_len = len(y)
		plt.savefig('../nn/'+str(x_len-1)+'.png')

	plt.pause(0.001)

import argparse
import os
if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-d','--meta',help='meta file')
	parser.add_argument('-m','--model',help='model path')
	parser.add_argument('-u','--muscle',help='muscle path')
	parser.add_argument('-a','--adaptive',help='adaptive sampling path')

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

	if args.adaptive is not None:
		if ppo.use_adaptive_sampling is False:
			print("Dont put : -a command")
			sys.exit()
		else:
			ppo.LoadModel_Marginal(args.adaptive)
	else:
		if ppo.use_adaptive_sampling:
			ppo.SaveModel_Marginal()

	print('num states: {}, num actions: {}'.format(ppo.env.GetNumState(),ppo.env.GetNumAction()))

	for i in range(ppo.max_iteration-5):
		ppo.Train()
		rewards = ppo.Evaluate()
		if (i%1000 == 0) or (i == ppo.max_iteration-7):
			Plot(rewards,'reward',0,False,True)
		else:
			Plot(rewards,'reward',0,False,False)


