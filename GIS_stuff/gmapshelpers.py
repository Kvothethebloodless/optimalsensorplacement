from __future__ import division

import numpy as np
import pdb

__TILE_SIZE = 256
__INV_TILE_SIZE = 1 / __TILE_SIZE
TILE_SIZE = __TILE_SIZE  # So that external functions can read, but not write to this variable's value.
MAX_ZOOM = 23
D2R = np.pi / 180
R2D = 180 / np.pi
PI4 = np.pi * 4
INV_PI4 = 1 / PI4
EXP2 = np.array([np.power(2, i) for i in range(0, 32)])
INVEXP2 = np.array([np.power(np.float(2), -i) for i in range(0, 32)], dtype='float32')

# todo: Add documentation. Make some functions publicly available.




def latlngToWorld((lat, lon)):
    # pdb.set_trace()
    #Copied from https://developers.google.com/maps/documentation/javascript/examples/map-coordinates
    siny = np.sin(lat*np.pi/180)
    siny = np.min(np.max(siny, -0.9999), 0.9999)
    mercpoint = np.array((0.5 + (lon / 360), 0.5 - ((np.log((1 + siny) / (1 - siny))) * (INV_PI4))), dtype=float)  # print mercpoint*__TILE_SIZE
	return mercpoint * __TILE_SIZE
    #Works!

def latlngToPixel((lat,lon),z):

    #Returns the pixel coordinate of a certain point at the required zoom level.
    (worldX,worldY) = latlngToWorld((lat,lon))
    (pixelX,pixelY) = worldToPixel((worldX,worldY),z)
    return (pixelX,pixelY)


def latlngToTile((lat,lng),z):
    #Returns cordinates of the tile involving the latitude and longitude at that zoom level
    (worldX,worldY) = latlngToWorld((lat,lng))
    (pixelX,pixelY) = worldToPixel((worldX,worldY),z)
    (tileX,tileY) = pixelToTile((pixelX,pixelY))
    return (tileX,tileY)

def pixelToTile((pixelX,pixelY)):
    _validrequest = True 
    if _validrequest:
		tileX = np.floor(pixelX * __INV_TILE_SIZE)
		tileY = np.floor(pixelY * __INV_TILE_SIZE)
        return np.asarray((tileX,tileY),dtype=int)
    else:
        raise ValueError('Invalid zoom value. Must be under 21.')


def worldToPixel((worldX,worldY),z):
    _validrequest = False
    _validrequest = True if z<=MAX_ZOOM else False
    if _validrequest:
        xcord = np.floor(worldX * EXP2[z])
        ycord = np.floor(worldY * EXP2[z])
        return np.asarray((xcord,ycord),dtype=int)
    else:
        raise ValueError('Invalid zoom value. Must be under 21.')

def tileToPixel((tileX,tileY)):
   #Gives top left corner pixel value. 
   pixelX = tileX * __TILE_SIZE
   pixelY = tileY * __TILE_SIZE
    return (pixelX,pixelY)


def tileToCenterPixel((tileX,tileY)):
   # Returns center pixel of tile
   (pixelX, pixelY) = tileToPixel((tileX, tileY))
   return (pixelX + 128, pixelY + 128)


def pixelToWorld((pixel_x,pixel_y),z):
    #Converts pixel coordinates at a zoom level to world coordinates
    # Vectorized already anyway.
    # pdb.set_trace()
	z = np.array(z).astype('int')
    if np.any(np.abs(np.array((pixel_x, pixel_y))) > EXP2[z + 8]):
        raise ValueError('Invalid Pixel value for given zoom')

    scale_factor = INVEXP2[z]
    world_x = np.asarray(pixel_x, dtype='float') * scale_factor
    world_y = np.asarray(pixel_y, dtype='float') * scale_factor
    return (world_x,world_y)


def pixelToLatlng_corner((pixelX, pixelY), z):
	# latlong coordinate of a pixel's top left corner.
    (worldX,worldY) = pixelToWorld((pixelX,pixelY),z)
    return worldToLatLon((worldX,worldY))


def tileToLatlng((tileX, tileY), z):
    if np.any(np.array((tileX, tileY) > EXP2[z + 8])):
        raise ValueError('Invalid tile value for the given zoom')
    centrepixel = tileToCenterPixel((tileX,tileY))
	latlng = pixelToLatLng_center(centrepixel, z)
    print ('Latitude and Longitude of the centre pixel %s are %s')%(centrepixel,latlng)
    return latlng


def pixelToLatLng_center(pixelXY, zoom):
	"""
	:param pixelXY: expects a nby2 numpy array or a single tuple of lat,lng in that order.
	:param z: expects a integer zoom value, of range 0-MAX_ZOOM
	:return: returns the latitude, longitude of the center of the pixel at the specified zoom level
			 in the same data object type they arrived in. Tuple->Tuple, or Array->Array.
	.. seealso:: pixelToLatLng_corner()
	"""

    #For better calulcation. By default, the pixeltoLatLng converts to the Lat Long 
    # corresponding to the top righ corner of the pixel. So for better result, we are
    #infact calculating all the four sides (Different pixels corners, infact) and
    # averaging them up.
    #pdb.set_trace()

	z = np.array(zoom).astype(
		'int')  # Just to make it a int. If it is a single number, it can still be used as a scalar
	# after converting it to an array. It's alright!
	try:
		if isinstance(pixelXY, tuple):
			pixelX = pixelXY[0]
			pixelY = pixelXY[1]

		elif hasattr(pixelXY, '__iter__'):
			# It is very likely a numpy array or a list of lists.
			pixelX = np.array(pixelXY[:, 0]).astype('int')
			pixelY = np.array(pixelXY[:, 1]).astype('int')

			if np.size(zoom) == 1:
				zoom = np.ones(np.size(pixelX)) * zoom
			elif np.size(zoom) != np.size(pixelX):
				raise RuntimeError("Invalid number of zoom values. Does'n't match the size of Pixel Array.")



	except TypeError:
		raise ('Input must be a tuple or a numpy array.')

	pdb.set_trace()
	w = pixelToWorld((pixelX, pixelY), z)
	list_sides = []
	list_sides.append(w)
	list_sides.append(pixelToWorld((pixelX, pixelY + 1), z))
	list_sides.append(pixelToWorld((pixelX + 1, pixelY), z))
	list_sides.append(pixelToWorld((pixelX + 1, pixelY + 1), z))
	ele1 = 0
	ele2 = 0
	for i in range(4):
		# print(list_sides[i])
		ele1 += list_sides[i][0]
		ele2 += list_sides[i][1]
	w_new = (ele1 / 4, ele2 / 4)
	# print w_new
	# Vectorized already.
	latlng_final = worldToLatLon(w_new)
	# print(latlng_final)
	return (latlng_final)




def worldToLatLon((world_x,world_y)):
    #    pdb.set_trace()
    # Is vectorized already! Awaiting performance tests.
	lng = ((world_x * __INV_TILE_SIZE) * 360) - 180
	p = np.exp((-(world_y * __INV_TILE_SIZE) + .5) * PI4)
    lat = np.float(R2D) * (np.arcsin((p - 1) / (1 + p)))
    return ((lat,lng))


