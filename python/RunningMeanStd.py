import numpy as np

class RunningMeanStd(object):
	def __init__(self, shape=()):
		self.mean = np.zeros(shape, 'float32')
		self.var = np.ones(shape, 'float32')
		self.count = 1e-4
		self.epsilon = 1e-8
		self.clip = 10

	def set(self, mean, var, count):
		self.mean = mean
		self.var = var
		self.count = count

	def save(self, path):
		print('save rms ../nn/{}_mean.npy'.format(path))
		np.save('../nn/'+path+'_mean.npy', self.mean)
		print('save rms ../nn/{}_var.npy'.format(path))
		np.save('../nn/'+path+'_var.npy', self.var)

	def load(self, path):
		print('load rms ../nn/{}_mean.npy'.format(path))
		self.mean = np.load('../nn/'+path+'_mean.npy')
		print('load rms ../nn/{}_var.npy'.format(path))
		self.var = np.load('../nn/'+path+'_var.npy')
		self.count = 200000000

	def load2(self, path):
		path = path[:-3]
		print("path : ", path)
		print('load rms {}_mean.npy'.format(path))
		self.mean = np.load(path+'_mean.npy')
		print('load rms {}_var.npy'.format(path))
		self.var = np.load(path+'_var.npy')
		self.count = 223

	def update(self, x):
		batch_mean = np.mean(x, axis=0)
		batch_var = np.var(x, axis=0)
		batch_count = x.shape[0]
		self.update_from_moments(batch_mean, batch_var, batch_count)

	def update_from_moments(self, batch_mean, batch_var, batch_count):
		self.mean, self.var, self.count = update_mean_var_count_from_moments(
			self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

	def apply(self, x, update=True):
		if update:
			self.update(x)
		x = np.clip((x - self.mean) / np.sqrt(self.var + self.epsilon), -self.clip, self.clip)
		return x

	def apply_no_update(self, x):
		x = np.clip((x - self.mean) / np.sqrt(self.var + self.epsilon), -self.clip, self.clip)
		return x

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

