from Model import *

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from matplotlib.widgets import Button
from pymss import EnvManager

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

class Graph(object):
	def __init__(self, env, dim, path):
		self.domain = [0,2]
		self.dim = len(self.domain)
		self.count = 0
		self.n = 50 # resolution
		self.path = path

		if self.dim == 1:
			fig = plt.figure()
			self.ax = fig.add_subplot(1,1,1)
		else:
			fig = plt.figure()
			self.ax = fig.gca(projection='3d')
			self.ax.zaxis.set_major_locator(LinearLocator(10))
			self.ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

		# Domain Range
		self.lo = [env.GetMinV()[2], env.GetMinV()[3]]
		self.hi = [env.GetMaxV()[2], env.GetMaxV()[3]]

		self.X = []
		for i in range(self.dim):
			self.X.append(np.arange(self.lo[i], self.hi[i], (self.hi[i]-self.lo[i])/float(self.n)))

		self.X[0], self.X[1] = np.meshgrid(self.X[0], self.X[1])

		self.marginal_model = MarginalNN(5)
		if use_cuda:
			self.marginal_model.cuda()

		if self.dim == 1:
			self.V = np.zeros(self.n)
		elif self.dim == 2:
			self.V = np.zeros((self.n, self.n))

		# for i in range(1,4):
		# 	path = '../nn/'+str(i*5)+'_marginal.pt'
		# 	self.ax.clear()
		# 	self.loadandplot(path)
		# 	plt.savefig('../nn/m_'+str(i*5)+'.png')

		self.loadandplot(self.path)

	def loadandplot(self, path):
		self.marginal_model.load(path)
		if self.dim == 2:
			for i in range(self.n):
				for j in range(self.n):
					state = [1.0, 0.6, self.X[0][i][j], self.X[1][i][j], 0.3]
					state = FloatTensor(np.array(state))
					self.V[i][j] = self.marginal_model(state).cpu().detach().numpy()
			self.ax.set_zlim(0.00, np.max(self.V))
			# self.ax.set_zlim(0.00, 12.0)
			self.surf = self.ax.plot_surface(self.X[0], self.X[1], self.V, linewidth = 0, antialiased = False)
		else:
			for i in range(self.n):
				state = FloatTensor(np.array([self.X[i]]))
				self.V[i] = self.marginal_model(state).cpu().detach().numpy()
			self.ax.plot(self.X, self.V)

	def next(self, event):
		self.count += 1
		self.ax.clear()
		path = '../nn/'+str(self.count)+'_marginal.pt'
		self.loadandplot(path)

	def prev(self, event):
		self.count -= 1
		if self.count < 0:
			self.count = 0
		self.ax.clear()
		path = '../nn/'+str(self.count)+'_marginal.pt'
		self.loadandplot(path)

import argparse
if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('-m', '--marginal_model', help='model path')
	parser.add_argument('-d', '--meta', help = 'meta file')

	args = parser.parse_args()

	if args.meta == None or args.marginal_model == None:
		print('Wrong Arguments')
		exit()

	path = args.marginal_model
	env = EnvManager(args.meta, 1)
	dim = env.GetNumParamState()

	callback = Graph(env, dim, path)

	bNextAxes = plt.axes([0.7, 0.05, 0.1, 0.075])
	bNext = Button(bNextAxes, 'Next')
	bNext.on_clicked(callback.next)

	bPrevAxes = plt.axes([0.8, 0.05, 0.1, 0.075])
	bPrev = Button(bPrevAxes, 'Prev')
	bPrev.on_clicked(callback.prev)

	plt.show()

