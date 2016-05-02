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
from tvtk.api import tvtk
from mayavi import mlab
import cv2


def scale(array,maxval, minval):
    array = array.astype('float32')
    maxval = float(maxval)
    minval = float(minval)
    array2 = array-array.min()
    array2 /= array2.max()
    print array2.max()
    array2 *= (maxval-minval)
    return array2+minval

class visualize():
	def __init__(self,(x,y,z),imagename,(xmin,xmax),(ymin,ymax)):
		#xmin,xmax,ymin,ymax in coordinates. Not, in real world numbers.
		self.X = x
		self.Y = y
		self.Z = z
		self.Image = cv2.imread(imagename)
		self.x = self.X[ymin:ymax,xmin:xmax]
		self.y = self.Y[ymin:ymax,xmin:xmax]
		self.z = self.Z[ymin:ymax,xmin:xmax]
		self.image = self.Image[ymin:ymax,xmin:xmax]
		self.

	def drawpoints(drawobj,points):
		size = 1 #square size is 3*2+1 = 7 units
		#points needs to be list of tuples
	#     pdb.set_trace()
		points_topleft = [tuple(point-[size,size]) for point in points]
		points_topright = [tuple(point-[size,-size]) for point in points]
		points_bottomleft = [tuple(point+[size,-size]) for point in points]
		points_bottomright = [tuple(point+[size,size]) for point in points]
		for i in range(len(points)):
	#         print(i)
			drawobj.polygon([points_topleft[i],points_bottomleft[i],points_bottomright[i],points_topright[i]],fill="red", outline="red")
		im.save('drawtest.jpeg')
		del drawobj

	def drawpoints(self,points):
