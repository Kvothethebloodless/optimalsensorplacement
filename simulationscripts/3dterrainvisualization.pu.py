"""
Draping an image over a terrain surface
"""
from tvtk.api import tvtk
from mayavi import mlab
import numpy as np
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

bmp1 = tvtk.JPEGReader()
bmp1.file_name="simulationdata/test2.jpg" #any jpeg file

image = misc.imread("simulationdata/test2.jpg")


my_texture=tvtk.Texture()
my_texture.interpolate=0
my_texture.set_input(0,bmp1.get_output())


# mlab.figure(size=(640, 800), bgcolor=(0.16, 0.28, 0.46))

a = np.load('simulationdata/corrected_terrain_mt.npy')

x = a[0]
y = a[1]
y = y.max()-y
z = scale(a[2],1000,1)

surf = mlab.mesh(x,y,z,color=(1,1,1))
surf.actor.enable_texture = True
surf.actor.tcoord_generator_mode = 'plane'
surf.actor.actor.texture = my_texture

mlab.show()