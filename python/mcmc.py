import numpy as np

# target_dist (target_distribution) f : x(state) -> double(probability)
# proposed_dist (proposed_distribution) f : x(current_state) -> x'(new_state)
# for sampling, object(MetropolisHasting).get_sample f : int(size) -> list(sampled values)

class MetropolisHasting:
	def __init__(self, size, min_v, max_v, target_dist, proposed_dist):
		if min_v.size != size or max_v.size != size:
			raise ValueError("[MetropolisHasting] Min != Max != Size")

		self.min_v = min_v
		self.max_v = max_v
		self.cur_x = np.ndarray(size)
		self.size = size
		self.target_dist = target_dist
		self.proposed_dist = proposed_dist
		self.result = []
		self.reset()

	def reset(self):
		self.cur_x = self.proposed_dist(self.cur_x, self.min_v, self.max_v)
		self.result = [self.cur_x]

	def compute_alpha(self, x_new, x_cur):
		v_cur = self.target_dist(x_cur)
		v_new = self.target_dist(x_new)
		if v_cur == 0:
			return 1.0

		return min(1.0, v_new/v_cur)

	def sample(self):
		x_new = self.proposed_dist(self.cur_x, self.min_v, self.max_v)
		alpha = self.compute_alpha(x_new, self.cur_x)

		if np.random.rand() <= alpha:
			self.cur_x = x_new

		self.result.append(self.cur_x)

	def get_sample(self, iter):
		for _ in range(iter):
			self.sample()

		return self.result
