from __future__ import division

import numpy as np
from scipy import interpolate
from scipy.integrate import quad
from scipy.optimize import fsolve as fs

import matplotlib.pyplot as plt


class terrain():
	def __init__(self,(x,y,z)):
		"""
		:parameter x: is the matrix of x spanning the terrain in the co-ordinate system with y axis flipped.(Described in detail in the doc)
		:parameter y: matrix of y corresponding to the above
		:parameter z: matrix of z corresponding to the above
		"""
		self.x = x
		self.y = y
		self.z = z
		self.height,self.width = self.x.shape
		self.fit2Dspline()
		self.inversefit2Dspline()

	def fit2Dspline(self):
		self.terrainsplinefit = interpolate.RectBivariateSpline(self.x[0],self.y[:,0],self.z)
		self.z_func = self.terrainsplinefit.ev #Works with vector inputs too.

	def inversefit2Dspline(self):
		x_indexes,y_indexes = np.mgrid[0:self.height,0:self.width]
		self.xindexsplinefit = interpolate.RectBivariateSpline(self.x[0],self.y[:,0],x_indexes)
		self.yindexsplinefit = interpolate.RectBivariateSpline(self.x[0],self.y[:,0],y_indexes)
		self.xindexfunc = self.xindexsplinefit.ev
		self.yindexfunc = self.yindexsplinefit.ev



	def subregion(self,(xmin,xmax),(ymin,ymax)):
		new_x = self.x[ymin:ymax,xmin:xmax]
		new_y = self.y[ymin:ymax,xmin:xmax]
		new_z = self.z[ymin:ymax,xmin:xmax]
		return (new_x,new_y,new_z)

	def realtocords(self,realpoints):
		xs = self.xindexfunc(realpoints[:,0],realpoints[:,1])
		ys = self.yindexfunc(realpoints[:,0],realpoints[:,1])
		cords = np.array([xs.astype(int), ys.astype(int)])
		return cords.T.reshape(-1, 2)



class road(terrain):
	def __init__(self,(road_x_cords,road_y_cords),(x,y,z)):
		"""
		:parameter road_x: is the set of indices of points on the x-axis for the interpolation
		:parameter road_y: is the set of indices points on the y-axes for the interpolation
		:parameter (x,y,z): The coordinates describing the full terrain in the real world system.
		:return:
		"""

		terrain.__init__(self,(x,y,z))
		# for i in range(len(road_x_cords)):

		self.road_y = y[road_y_cords,road_x_cords]
		self.road_x = x[road_y_cords,road_x_cords]
		# self.darray_to_x = np.vectorize(self.d_to_x)
		# self.xarray_to_d = np.vectorize(self.x_to_d)
		self.fit1Dspline()
		self.maxdistance = self.x_to_d(self.road_x[0],self.road_x[-1])
		self.startx = self.road_x[0]
		self.endx = self.road_x[-1]
		self.dtoxsplinefit()

	def plotroad(self):
		dw = np.mgrid[10:int(self.maxdistance),-3:3].T.reshape(-1,2)
		points = self.dwtoxyz(dw[:,0],dw[:,1],self.startx)
		plt.scatter(points[0],points[1])
		plt.show()

	def fit1Dspline(self):

		self.tck = interpolate.splrep(self.road_x,self.road_y,s=0)
		# if ler<0:
		# 	print "Fitting successful"

	def x_to_d(self,x1,x2):
		I = quad(lambda x:np.sqrt(1+(np.power(interpolate.splev(x,self.tck,der=1),2))),x1,x2)[0] #Works with vector inputs too
		return I
	# def pointstolength(self,x2_array,x1_array):


	def dtoxsplinefit(self):
		"""
		Instead of solving for the xtod for getting x given d, it seems
		a better idea to actually fit a curve from d to x. We are going to
		do that now.
		"""
		# pdb.set_trace()
		# self.roadx_extrapoints = np.arange(self.startx,self.endx,.5)
		# self.d_all = self.xarray_to_d(self.startx,self.roadx_extrapoints)
		#Decrease instantiation time by saving these to file.
		(self.d_all, self.roadx_extrapoints) = np.load('simulationdata/dtoxall.npy')
		self.dtox_tck = interpolate.splrep(self.d_all,self.roadx_extrapoints,s=0)

	def d_to_x(self,d):
		return (interpolate.splev(d,self.dtox_tck))

	def darray_to_x(self,darray,start_x): #Bad idea aagain
		l = []
		nele = darray.shape[0]
		for i in range(nele):
			l.append(self.d_to_x(start_x,darray[i]))
		return np.array(l)
	def xarray_to_d(self,x1,x2_array): #Decent Idea. Needs improvement.
		nele = x2_array.shape[0]
		l = []
		for i in range(nele):
			l.append(self.x_to_d(x1,x2_array[i]))
		return np.array(l)



	def d_to_x_old(self,length,start_x): #Bad idea! The solver is calling splev more than required.
 		x_l = fs(lambda x: self.x_to_d(start_x,x) - length,self.road_x[0])
		return x_l

	def dwtoxyz(self,d,w):
		# pdb.set_trace()
		x = self.d_to_x(d)

		y = interpolate.splev(x,self.tck,der=0)
		fprime = (interpolate.splev(x,self.tck,der=1))
		costheta = 1/np.sqrt(1+(np.power(fprime,2)))
		new_x = x - (((fprime)*w)*costheta)
		new_y = y + (w*costheta)
		new_z = self.z_func(new_x,new_y)
		return (new_x,new_y,new_z)
	# def dwtoxyz_array(self,darray,warray,start_x):
	# 	pdb.set_trace()
	# 	x = self.darray_to_x(darray,start_x)
	#
	# 	y = interpolate.splev(x,self.tck,der=0)
	# 	fprime = (interpolate.splev(x,self.tck,der=1))
	# 	costheta = 1/np.sqrt(1+(np.power(fprime,2)))
	# 	new_x = x - (((fprime)*warray)*costheta)
	# 	new_y = y + (warray*costheta)
	# 	new_z = self.z_func(new_x,new_y)
	# 	return (new_x,new_y,new_z)


















