from __future__ import division
import numpy as np
import os
from scipy import interpolate
from scipy.optimize import fsolve as fs
from scipy.integrate import quad
import pdb
import matplotlib.pyplot as plt
import solvers.psosolver as psos  # Pso solver
import utilities.statelogger as stlog  # Logging states and state variables
from tvtk.api import tvtk
from mayavi import mlab
import Image, ImageDraw
from scipy import misc


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
		pdb.set_trace()
		self.X = x
		self.Y = y
		self.Z = z
		print imagename
		self.imagename = imagename
		self.imageobject = Image.open(self.imagename)
		self.Image = misc.imread(imagename)
		self.x = self.X[ymin:ymax,xmin:xmax]
		self.y = self.Y[ymin:ymax,xmin:xmax]
		self.z = self.Z[ymin:ymax,xmin:xmax]
		self.image = self.Image[ymin:ymax,xmin:xmax]

	def converttomayavi(self,convertedfilename,imageobject):
		#MAYAVI seems to rotate 90 by counterclockwise and then flipleftright. So we do the opposite. We, flipleftright and ro
		#tate 90 by clockwise.
		# a = image
		# plt.imshow(a)
		# plt.show()
	#     r = a[:,:,0]
	#     g = a[:,:,1]
	#     b = a[:,:,2]
	#     rnew = np.fliplr(r.T)
	#     gnew = np.fliplr(g.T)
	#     bnew = np.fliplr(b.T)
	#     newimg = np.hstack((rnew.
	#     re = np.vstack((rnew.reshape(-1),gnew.reshape(-1), bnew.reshape(-1)))
	#     newimg = re.T.reshape((1024,1024,3))
	# 	plt.imshow(np.array(imageobject))
	# 	plt.show()
		newimg = imageobject.transpose(Image.FLIP_LEFT_RIGHT)
		# plt.imshow(np.array(newimg))
		# plt.show()
		# plt.figure()
		newimg = newimg.rotate(-90) #Clockwise rotation.
		# plt.imshow(np.array(newimg))
		# plt.show()
		convertedfilename = 'simulationdata/visualize/%smayavi.jpeg' %convertedfilename
		newimg.save(convertedfilename)

	def drawpoints(self,points,outname='default_drawpoints'):
		# im = Image.open('simulationdata/visualize/texture2.jpeg')
		im = self.imageobject
		drawobj = ImageDraw.Draw(im)
		size = 3 #square size is 3*2+1 = 7 units
		#points needs to be list of tuples
	#     pdb.set_trace()
		points_topleft = [tuple(point-[size,size]) for point in points]
		points_topright = [tuple(point-[size,-size]) for point in points]
		points_bottomleft = [tuple(point+[size,-size]) for point in points]
		points_bottomright = [tuple(point+[size,size]) for point in points]
		for i in range(len(points)):
	#         print(i)
			drawobj.polygon([points_topleft[i],points_bottomleft[i],points_bottomright[i],points_topright[i]],fill="red", outline="red")
		outfilename = "simulationdata/visualize/%spoints.jpeg" %outname
		im.save(outfilename)
		new_img = np.array(im)
		plt.imshow(new_img)
		del drawobj
		return im

	def showin3d(self,type,imagename):
		pdb.set_trace()
		bmp1 = tvtk.JPEGReader()
		bmp1.file_name="simulationdata/visualize/"+imagename +"mayavi.jpeg" #any jpeg file
		my_texture=tvtk.Texture()
		my_texture.interpolate=0
		my_texture.set_input(0,bmp1.get_output())
		# mlab.figure(size=(640, 800), bgcolor=(0.16, 0.28, 0.46))
		# a = np.load('simulationdata/corrected_terrain_mt.npy')
		x = self.X
		y = self.Y
		y = y.max()-y
		z = scale(self.Z,1000,1)
		surf = mlab.mesh(x,y,z,color=(1,1,1))
		surf.actor.enable_texture = True
		surf.actor.tcoord_generator_mode = 'plane'
		surf.actor.actor.texture = my_texture
		mlab.show()
		# if type = 'altitude':

