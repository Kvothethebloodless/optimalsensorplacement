from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import pdb

TILE_SIZE = 256
MAX_ZOOM = 23


def latlngToWorld((lat, lon)):
#     pdb.set_trace()
    #Copied from https://developers.google.com/maps/documentation/javascript/examples/map-coordinates
    siny = np.sin(lat*np.pi/180)
    siny = np.min(np.max(siny,-0.9999),0.9999)
    mercpoint =  np.array((0.5 + (lon/360), 0.5 - ((np.log((1+siny)/(1-siny)))/(4*np.pi))),dtype=float)
    print mercpoint*TILE_SIZE
    return mercpoint*TILE_SIZE
    #Works!

def latlngToPixel((lat,lon),z):

    #Returns the pixel coordinate of a certain point at the required zoom level.
    (worldX,worldY) = latlngToWorld((lat,lon))
    (pixelX,pixelY) = worldToPixel((worldX,worldY),z)
    return (pixelX,pixelY)


def latlngToTile((lat,lng),z):
    #Returns cordinates of the tile involving the latitude and longitude at that zoom level
    pdb.set_trace()
    (worldX,worldY) = latlngToWorld((lat,lng))
    (pixelX,pixelY) = worldToPixel((worldX,worldY),z)
    (tileX,tileY) = pixelToTile((pixelX,pixelY))
    return (tileX,tileY)

def pixelToTile((pixelX,pixelY)):
    _validrequest = True 
    if _validrequest:
        tileX = np.floor(pixelX/TILE_SIZE)
        tileY = np.floor(pixelY/TILE_SIZE)
        return np.asarray((tileX,tileY),dtype=int)
    else:
        raise ValueError('Invalid zoom value. Must be under 21.')
        return


def worldToPixel((worldX,worldY),z):
    _validrequest = False
    _validrequest = True if z<=MAX_ZOOM else False
    if _validrequest:
        xcord = np.floor(worldX*np.power(2,z))
        ycord = np.floor(worldY*np.power(2,z))
        return np.asarray((xcord,ycord),dtype=int)
    else:
        raise ValueError('Invalid zoom value. Must be under 21.')
        return

def tileToPixel((tileX,tileY)):
   #Gives top left corner pixel value. 
    pixelX = tileX*TILE_SIZE
    pixelY = tileY*TILE_SIZE
    return (pixelX,pixelY)


def tileToCenterPixel((tileX,tileY)):
   # Returns center pixel of tile
    (pixelX,pixelY) = tileToPixel((tileX,tileY));
    return (pixelX+128,pixelY+128)


def pixelToWorld((pixel_x,pixel_y),z):
    #Converts pixel coordinates at a zoom level to world coordinates
    if np.any(np.abs(np.array((pixel_x,pixel_y)))>np.power(2,z+8)):
        raise ValueError('Invalid Pixel value for given zoom')
        return

    scale = np.float(np.power(2,z))
    world_x = np.float(pixel_x) / scale
    world_y = np.float(pixel_y) / scale
    return (world_x,world_y)


def pixelToLatlng((pixelX,pixelY),z):
    (worldX,worldY) = pixelToWorld((pixelX,pixelY),z)
    return worldToLatLon((worldX,worldY))

def tileToLatlng((tileX,tileY),z):
    if np.any(np.array((tileX,tileY)>np.power(2,z))):
        raise ValueError('Invalid tile value for the given zoom')
        return
    centrepixel = tileToCenterPixel((tileX,tileY))
    latlng = pixelToLatlng_better(centrepixel,z)
    print ('Latitude and Longitude of the centre pixel %s are %s')%(centrepixel,latlng)
    return latlng



def pixelToLatlng_better((pixelX,pixelY),z):
    #For better calulcation. By default, the pixeltoLatLng converts to the Lat Long 
    # corresponding to the top righ corner of the pixel. So for better result, we are
    #infact calculating all the four sides (Different pixels corners, infact) and
    # averaging them up.
    w = pixelToWorld((pixelX,pixelY),z)
    list_sides = []
    list_sides.append(w)
    list_sides.append(pixelToWorld((pixelX, pixelY+1),z))
    list_sides.append(pixelToWorld((pixelX+1, pixelY),z))
    list_sides.append(pixelToWorld((pixelX+1, pixelY+1),z))
    ele1 = 0
    ele2 = 0
    for i in range(4):
        print(list_sides[i])
        ele1 += list_sides[i][0]
        ele2 += list_sides[i][1]
    w_new  = (ele1/4,ele2/4)
    print w_new
    latlng_final = worldToLatLon(w_new)
    print(latlng_final)
    return (latlng_final) 



def worldToLatLon((world_x,world_y)):
    lng = ((world_x/TILE_SIZE) - 0.5)*360
    p = np.exp((-(world_y/TILE_SIZE)+.5)*np.pi*4)
    print p
    print ((1-p)/(1+p))
    lat = np.float(180/np.pi)*(np.arcsin((p-1)/(1+p)))
    return ((lat,lng))


