from __future__ import division
import numpy as np
import os
from scipy import interpolate
from scipy.optimize import fsolve as fs
from scipy.integrate import quad
import pdb

import matplotlib.pyplot as plt

import src.arena as arena
import solvers.psosolver as psos  # Pso solver
import utilities.statelogger as stlog  # Logging states and state variables

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

	def fit2Dspline(self):
		self.terrainsplinefit = interpolate.RectBivariateSpline(self.x[0],self.y[:,0],self.z)
		self.z_func = self.terrainsplinefit.ev #Works with vector inputs too.

	def subregion(self,(xmin,xmax),(ymin,ymax)):
		new_x = self.x[ymin:ymax,xmin:xmax]
		new_y = self.y[ymin:ymax,xmin:xmax]
		new_z = self.z[ymin:ymax,xmin:xmax]
		return (new_x,new_y,new_z)


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
		self.darray_to_x = np.vectorize(self.d_to_x)
		self.xarray_to_d = np.vectorize(self.x_to_d)
		self.fit1Dspline()

	def fit1Dspline(self):

		self.tck = interpolate.splrep(self.road_x,self.road_y,s=0)
		# if ler<0:
		# 	print "Fitting successful"

	def x_to_d(self,x1,x2):
		I = quad(lambda x:np.sqrt(1+(np.power(interpolate.splev(x,self.tck,der=1),2))),x1,x2)[0] #Works with vector inputs too
		return I
	# def pointstolength(self,x2_array,x1_array):


	def d_to_x(self,length,start_x=0):
		x_l = fs(lambda x: self.x_to_d(start_x,x) - length,0.0)
		return x_l

	def dwtoxyz(self,d,w,start_x=0):
		pdb.set_trace()
		x = self.darray_to_x(d,start_x)

		y = interpolate.splev(x,self.tck,der=0)
		fprime = (interpolate.splev(x,self.tck,der=1))
		costheta = 1/np.sqrt(1+(np.power(fprime,2)))
		new_x = x - (((fprime)*w)*costheta)
		new_y = y + (w*costheta)
		new_z = self.z_func(new_x,new_y)
		return (new_x,new_y,new_z)


















