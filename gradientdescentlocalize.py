import numpy as np
import matplotlib.pyplot as plt	
#import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

class gradientdescentsolver():
	
	def __init__(self,score_func,gradient_score_func,sensor_arena):
		self.score_func = score_func;
		self.gradient_func = gradient_score_func
		self.accel_coeff = .4
		self.posseries_data =[] #np.empty(np.shape(self.current_pos))
		self.centroidseries_data = [0]#np.empty(self.no_dim);
		self.spreadseries_data = [0];
		self.sensor_arena = sensor_arena
		self.score_data =[]
	def compute_gradient(self,point):
		self.grad = self.gradient_func(point)
	def descent(self,point):
		self.new_point = point - self.grad*self.accel_coeff
	def get_descented_point(self,point):
		self.point =point
		self.compute_gradient(point);
		self.descent(point)
		return self.new_point
	def report(self):
		print("The solver is currently at "+ str(self.point) +"\n")
		print(self.sensor_arena.get_score(self.point))
		print(self.score_func(self.point))
		print("The current score is " + str(self.sensor_arena.get_score(self.point))+ "\n")
		print("The computed gradient is " + str(self.grad) + "\n")
		print("The point moved to " +str(self.new_point)+"\n")
		print("The new score is" + str(self.score_func(self.new_point))+"\n")
		
	def valueIO(self):	
		self.current_pos_dummy = np.copy(sensor_arena.sensor_loc)
		self.current_pos_dummy = np.vstack((self.current_pos_dummy,sensor_arena.target_loc))
		self.current_pos_dummy = np.vstack((self.current_pos_dummy,self.new_point))
		self.posseries_data.append(self.current_pos_dummy)
		self.spreadseries_data.append(1)
		self.score_data.append(self.score_func(self.point))
		#np.save('data_tracker.npy',[np.array(self.posseries_data
		np.save('data_tracker',[np.array(self.posseries_data),np.array(self.centroidseries_data),np.array(self.spreadseries_data),np.array(self.score_data)])
		
	def plot_score(self):	
		a = load('data_tracker.npy');
		plot(a[3])
			
		
		
	

class localize():
	
	def __init__(self,no_sensors):
		self.no_sensors = no_sensors
		self.create_sensors()
		self.create_target_object()
		self.get_original_ranges()
		self.get_noisy_ranges()
		
	def dist_from_ithsensor(self,i,point):
		if i>self.no_sensors:
			print "Exceeded"
			return
		else:
			return np.linalg.norm(point-self.sensor_loc[i])
	
	
	
	def gradient_score(self,point):
		point = np.array(point);
		dim = point.shape[0];
		gradi = np.empty(dim)
		dist_vector = [self.dist_from_ithsensor(i,point) for i in range(self.no_sensors)]
		dist_vector = np.array(dist_vector)
		common_factor_vector = [1-((self.noisy_ranges[i])/dist_vector[i]) for i in range(self.no_sensors)]
		common_factor_vector = np.array(common_factor_vector)
		dim_diff_vector = point-self.sensor_loc;
		dim_gradient_vector = np.transpose(common_factor_vector*np.transpose(dim_diff_vector));
		dim_gradient = np.sum(dim_gradient_vector,axis=0)
		return dim_gradient*(2./self.no_sensors)
		#grad_presum_vector = [np.dot(common_factor_vector[i],dim_diff_vector[i])
		
		#for i in range(	dim):		
			##gradi[dim] = 2*(self.noisy_ranges[i]/self.dist_from_ithsensor(i,)
			
		
	def create_sensors(self):
		#self.sensor_loc = np.random.random((self.no_sensors,2))*10
		#self.sensor_loc =np.array([[1,2],[3,4],[5,6]])
		self.sensor_loc = np.array([[0.969,.266],[.66,.41],[.52,.78]]) * 10
	def create_target_object(self):
		#self.target_loc = np.random.random((1,2))*10
		self.target_loc = [5,5]
		
	def get_original_ranges(self):
		self.orig_ranges = self.sensor_loc-self.target_loc;
		self.orig_ranges = np.linalg.norm(self.orig_ranges,axis=1)
	
	def get_noisy_ranges(self):
		sigma = .1;
		
		mean_vector = self.orig_ranges;
		path_loss_coeff = 0.01;
		variance_vector = (sigma)*(np.power(self.orig_ranges,path_loss_coeff));
		#print mean_vector
		#print variance_vector
		self.mean = mean_vector
		self.var = variance_vector
		nse = np.arange(self.no_sensors)
		for i in range(self.no_sensors):			
			nse[i] = np.random.normal(mean_vector[i],variance_vector[i])
		#nse = np.array(n)
		
		self.noisy_ranges = nse
	def get_score(self,particle_loc):
		score = 0;
		cartesian_distance = np.linalg.norm(particle_loc -self.sensor_loc, axis=1)
		#print cartesian_distance
		#cartesian_distance = np.power(cartesian_distance,.5)
		#print cartesian_distance
		score_vector = self.noisy_ranges-cartesian_distance;
		#print score_vector
		score = np.mean(np.power(score_vector,2))
		
		return score
	
		
		
		
	#def initalize_swarm_param(self):
		
def plot_surface(fitfunc):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	X1 = np.arange(0, 10, 0.25)
	Y2 = np.arange(0, 10, 0.25)
	X, Y = np.meshgrid(X1, Y2)
	Z = X1;
	for i in range(len(X)):
		Z[i] = fitfunc(np.array([X1[i],Y2[i]]))
	Z = np.array(Z)
	#R = np.sqrt(X**2 + Y**2)
	#Z = np.sin(R)
	surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
	#ax.set_zlim(-1.01, 1.01)

	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

	fig.colorbar(surf, shrink=0.5, aspect=5)

	plt.show()		
		
sensor_arena = localize(3)
gdsolver = gradientdescentsolver(sensor_arena.get_score,sensor_arena.gradient_score,sensor_arena)

point = np.random.random((1,2))
iterindx =1

while iterindx<500:
	curr_score = sensor_arena.get_score(point)
	point = gdsolver.get_descented_point(point)
	descented_score = sensor_arena.get_score(point)
	deltascore = descented_score-curr_score
	curr_score = descented_score;
	iterindx = iterindx+1
	gdsolver.report()
	gdsolver.valueIO()
	
