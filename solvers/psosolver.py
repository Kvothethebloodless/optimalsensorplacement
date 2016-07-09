import numpy as np
import pdb
from random import randint

from utilities import statelogger as stlog


#


# import matplotlib.pyplot as plt
# import tables as tb

def scale_array(arr, max_vl, min_vl):
	arr_min = np.min(np.array(arr))
	arr_max = np.max(np.array(arr))
	scaled_array = (arr - arr_min)
	scaled_array = scaled_array * (max_vl - min_vl)
	return scaled_array + min_vl


class PSO:
	def __init__(self, *args, **kwargs):


		"""
		kwargs needs the following arguments.

		no_particles=20, no_dim, self_accel_coeff=.5, global_accel_coeff=.5,one_point,
				 searchspaceboundaries,dt



		no_particles, no_dim, self_accel_coeff, global_accel_coeff, dt , A total of 5 parameters
		no_particles = Number of swarm particles to create and engage
		no_dim = Dimension of the search space
		self_accel_coeff = acceleration coefficient following its own past
		global_accel_coeff = accleration coefficient following the group leader
		dt_pos = dt for velocity to position conversion
		dt_velocity = dt for accleration to velocity conversion. Both should normally be equal.
		Searchboundaries = a 2byndim array with the first component being the inner boundary values and the second component being the outer boundary values

		maxvels = a 1byndim array with maximum allowed velocity ofthe particle in the dimension.

		datafilename = filename to store the pso simulation data. Appends npy automatically.
		loggerfilename = filename to log the status messages from the dataframelogger.



		"""


		# pdb.set_trace()
		self.score_func = args[0]
		# self.no_dim = args[1]
		self.no_particles = kwargs.get('no_particles',20)
		self.no_dim = args[1]


		print(kwargs)

		self.self_accel_coeff = kwargs.get('self_accel_coeff',.5)
		self.global_accel_coeff = kwargs.get('global_accel_coeff',.5)
		self.transform = kwargs.get('transform',False)
		self.spacetransformfunction = kwargs.get('spacetransform',lambda l: l) #If no transformation, make it a unitary transformation
		self.dt_velocity = kwargs.get('dt_velocity',1)
		self.dt_pos = kwargs.get('dt_pos',1)
		self.weight = kwargs.get('weight',.5)
		self.maxvel = kwargs.get('maxvel',10)
		self.minvel = kwargs.get('minvel',0)
		self.datalogfilename = kwargs.get('datalogfilename','datalog')
		self.datafilename = kwargs.get('datafilename','psodata')
		self.solverlogfilename = kwargs.get('solverlogfilename','psosolverlog')

		self.searchspaceboundaries = kwargs.get('searchspaceboundaries',np.random.random((2,self.no_dim)))

		self.maxvels = kwargs.get('maxvels',((np.abs(self.searchspaceboundaries[0]-self.searchspaceboundaries[1])).astype(float)/2))
		self.innerboundary = self.searchspaceboundaries[0]
		self.outerboundary = self.searchspaceboundaries[1]
		if self.no_dim == -1:
			raise Exception('No dimensions is a required parameter')

		self.initialize_swarm()

	def initialize_swarm(self):
		self.velocity = np.random.random((self.no_particles, self.no_dim))
		# self.current_pos = (np.random.random((self.no_particles, self.no_dim))) * 10 - (
		#     5 * np.ones((self.no_particles, self.no_dim)))
		# self.current_pos =
		list_pos = []  # An empty list to hold lists of positions of each particle in each dimension. i.e, if there are 10 dims and 50 particles, this list will
		# consist of 10 lists of 50 sub entries which specify the dimension of each particle in that dimension.
		pdb.set_trace()
		for i in range(self.no_dim):
			# print i
			ithdim_pos = np.random.uniform(self.innerboundary[i], self.outerboundary[i], self.no_particles)
			list_pos.append(ithdim_pos)

		pos_array = np.array(list_pos)
		self.current_pos = np.reshape(np.transpose(pos_array), (self.no_particles, self.no_dim))

		# self.paramspacebounds = [-10, 10]
		self.selfminlocation = self.current_pos  # np.random.random((self.no_particles,self.no_dim));
		self.selfminval = self.funcdef(self.selfminlocation, self.no_particles)
		self.globalmin = np.min(self.selfminval)
		self.globalminlocation = self.selfminlocation[np.argmin(self.selfminval)]
		self.curr_score = self.funcdef(self.current_pos, self.no_particles)
		self.centroid = np.mean(self.current_pos, axis=0)
		self.spread = 0
		print ('Initialized Swarm Locations are:')

		print (self.current_pos)

	def update_selfmin(self):
		update_req_array = self.curr_score < self.selfminval

		self.selfminval = np.multiply(update_req_array, self.curr_score) + np.multiply(np.invert(update_req_array),
																					   self.selfminval)

		update_req_array = np.transpose(update_req_array)

		update_req_array = np.reshape(update_req_array, (self.no_particles, 1))
		# print(np.shape(update_req_array));

		# print(np.shape(self.current_pos));
		self.selfminlocation = (update_req_array * self.current_pos) + (np.invert(update_req_array) * self.selfminlocation)

	def update_globalmin(self):
		curr_globalmin = np.amin(self.curr_score)
		curr_globalmin_loc = self.current_pos[self.curr_score == curr_globalmin]
		curr_globalmin_loc = curr_globalmin_loc[0]
		state = curr_globalmin < self.globalmin
		state = np.squeeze(state)
		self.globalmin = np.multiply(state, curr_globalmin) + np.multiply(np.invert(state), self.globalmin)
		self.globalminlocation = (state * curr_globalmin_loc) + (np.invert(state) * self.globalminlocation)

	def funcdef(self, locs, no_particles):
		score_curr = np.empty(no_particles)

		for i in range(no_particles):
			score_curr[i] = self.score_func(locs[i])
		# print(score_curr[i])
		return score_curr

	def solve_iters(self, no_iters, verbose=True, out=True, randomname=False):

		# pdb.set_trace()
		# self.psodatalogger = stlog.statelogger(self.datafilename,self.datalogfilename,self.current_pos,self.centroid,self.spread,self.globalmin)

		if randomname:
			randomname = str(randint(100000, 999999))
			self.psodatalogger = stlog.statelogger(randomname, self.datalogfilename, self.current_pos, self.centroid,
												   self.spread, self.globalmin)
		else:
			self.psodatalogger = stlog.statelogger(self.datafilename, self.datalogfilename, self.current_pos,
												   self.centroid, self.spread, self.globalmin)

		self.initialize_swarm()
		for i in range(no_iters):
			self.update_pos()
			self.update_currscores()
			self.update_selfmin()
			self.update_globalmin()
			self.update_velocities()


			if verbose:
				(centroid, spread) = self.calc_swarm_props()


				print('\n \n \n \n \n')
				print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n')
				print("The running index is " + str(i) + "\n")
				# print(str(psoslr.globalminlocation))
				self.report()
				print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n')
			# curr_score = curr_arena.get_score(point)
			self.psodatalogger.add_state(self.current_pos,self.centroid,self.spread,self.globalmin)
		# point = gdsolver.get_descented_point(point)
		# descented_score = curr_arena.get_score(point)
		# deltascore = descented_score - curr_score
		# curr_score = descented_score
		# gdsolver.report()
		# pdb.set_trace()

		self.psodatalogger.close_logger()
		if out:
			if randomname:
				output = dict([('extremum value', self.globalmin), ('extremum location', self.globalminlocation),
							   ('filename', randomname)])
			else:
				output = dict([('extremum value', self.globalmin), ('extremum location', self.globalminlocation)])

			return output
		else:
			return

	def solve_convergence(self, convergencelimit, iterbound=1000, verbose=False, out=True, checkiters=10,
						  randomname=True):

		if randomname:
			randomname = str(randint(100000, 999999))
			self.psodatalogger = stlog.statelogger(randomname, self.datalogfilename, self.current_pos, self.centroid,
												   self.spread, self.globalmin)
		else:
			self.psodatalogger = stlog.statelogger(self.datafilename, self.datalogfilename, self.current_pos,
												   self.centroid, self.spread, self.globalmin)

		self.initialize_swarm()
		iter = 0
		cur_score = self.globalmin
		# diff = np.inf
		errorlist = [np.inf for i in range(checkiters)]
		# pdb.set_trace()
		while ((np.any(np.abs(errorlist[-checkiters::]) > convergencelimit)) & (iter<iterbound) ):

			# iter = 0
			# cur_score = self.globalmin
			iter += 1
			self.update_pos()
			self.update_currscores()
			self.update_selfmin()
			self.update_globalmin()
			self.update_velocities()

			if verbose:
				(centroid, spread) = self.calc_swarm_props()
				print('\n \n \n \n \n')
				print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n')
				print("The running index is " + str(iter) + "\n")
				# print(str(psoslr.globalminlocation))
				self.report()
				print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n')
			# curr_score = curr_arena.get_score(point)

			self.psodatalogger.add_state(self.current_pos,self.centroid,self.spread,self.globalmin)
			diff = cur_score - self.globalmin
			errorlist.append(diff)
			cur_score = self.globalmin
		# point = gdsolver.get_descented_point(point)
		# descented_score = curr_arena.get_score(point)
		# deltascore = descented_score - curr_score
		# curr_score = descented_score
		# gdsolver.report()
		self.psodatalogger.close_logger()

		if iter == iterbound:
			print ('Iterations bound exceeded')
		if out:
			if randomname:
				output = dict([('extremum value', self.globalmin), ('extremum location', self.globalminlocation),
							   ('Number of iterations', iter), ('filename', randomname)])
			else:
				output = dict([('extremum value', self.globalmin), ('extremum location', self.globalminlocation),
							   ('Number of iterations', iter)])
			print errorlist
			return output
		else:
			return






	def update_currscores(self):
		self.curr_score = self.funcdef(self.current_pos, self.no_particles)

	def update_velocities(self):
		self_accel = np.random.random() * (self.selfminlocation - self.current_pos)
		global_accel = np.random.random() * (self.globalminlocation - self.current_pos)
		accel = self.self_accel_coeff * self_accel + self.global_accel_coeff * global_accel
		self.velocity = self.weight * self.velocity + accel * self.dt_velocity
		g_mask = self.velocity > self.maxvels
		l_mask = self.velocity < -self.maxvels
		g_mask = np.array([np.all(ele) for ele in g_mask])
		l_mask = np.array([np.all(ele) for ele in l_mask])
		self.velocity[g_mask] = self.maxvel
		self.velocity[l_mask] = -self.maxvel

	def update_pos(self):
		self.current_pos += self.velocity * self.dt_pos
		self.current_pos = self.searchspacelimit(self.current_pos)

	def searchspacelimit(self, pos):
		for i in range(self.no_dim):
			pos[pos[:, i] > self.outerboundary[i], i] = self.outerboundary[i]
			pos[pos[:, i] < self.innerboundary[i], i] = self.innerboundary[i]
		return pos

	def calc_swarm_props(self):
		centroid = np.mean(self.current_pos, axis=0)
		spread = 0
		for i in range(self.no_particles):
			spread += np.linalg.norm(self.current_pos - centroid)
		self.centroid = centroid
		self.spread = spread
		self.best_score = np.min(self.curr_score)
		return centroid, spread




	def report(self):
		print("The centroid is at " + str(self.centroid) + "\n")
		print("The spread is " + str(self.spread) + "\n")
		print("------")
		print("The current best score  is" + str(np.min(self.curr_score)) + "\n")
		print("The global best is at " + str(self.globalminlocation) + "\n")
		print("The gloabl best score is" + str(self.globalmin) + "\n")
		# print("Ca")

		if self.no_particles < 20:
			print(
				"current score \t\t best self score\t\t current_position \t\t best self position \t\t current velocity \n")
			for i in range(self.no_particles):
				print(str(self.curr_score[i]) + '\t\t ' + str(self.selfminval[i]) + '\t\t ' + str(
						self.current_pos[i]) + '\t\t  ' + str(self.selfminlocation[i]) + '\t\t ' + str(
						self.velocity[i]))
			# print(self.current_pos[i])
			# print(self.velocity[i])
			# print("The current score is  "+str(self.curr_score)+"\n");
			# print("The positions are " + str(self.current_pos)+"\n");
			# print("-----");


			# print("The current velocities are "+ str(self.velocity)+"\n");
			print("#$#$#$#$#$#$ \n")
