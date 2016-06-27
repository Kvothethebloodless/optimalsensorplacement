from __future__ import division
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import pdb

TILE_SIZE = 256;
MAX_ZOOM = 23;
def latlngToPixel((lat,lon),zoom):
    #Returns the pixel coordinate of a certain point at the required zoom level.
    (worldX,worldY) = latlngToWorld((lat,lon))
    (pixelX,pixelY) = worldToPixel((worldX,worldY),zoom)
    return (pixelX,pixelY)

def latlngToWorld((lat,lon)):
#     pdb.set_trace()
    #Copied from https://developers.google.com/maps/documentation/javascript/examples/map-coordinates
    siny = np.sin(lat*np.pi/180)
    siny = np.min(np.max(siny,-0.9999),0.9999)
    mercpoint =  np.array((0.5 + (lon/360), 0.5 - ((np.log((1+siny)/(1-siny)))/(4*np.pi))),dtype=float)
    print mercpoint*TILE_SIZE
    return mercpoint*TILE_SIZE
    #Works!

def latlngToTile((lat,lng),zoom):
    #Returns cordinates of the tile involving the latitude and longitude at that zoom level
    pdb.set_trace()
    (worldX,worldY) = latlngToWorld((lat,lng))
    (pixelX,pixelY) = worldToPixel((worldX,worldY),zoom)
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
def tileToPixel((tileX,tileY)):
   #Gives top left corner pixel value. 
    pixelX = tileX*TILE_SIZE
    pixelY = tileY*TILE_SIZE
    return (pixelX,pixelY)
#def tileToCenterPixel((tileX,tileY)):
    #Returns center pixel of tile
#    (pixelX,pixelY) = tileToPixel(tileX,tileY);
#    return (pixelX+128,pixelY+128)



def worldToPixel((worldX,worldY),zoom):
    validrequest = False
    _validrequest = True if zoom<=MAX_ZOOM else False
    if _validrequest:
        xcord = np.floor(worldX*np.power(2,zoom))
        ycord = np.floor(worldY*np.power(2,zoom))
        return np.asarray((xcord,ycord),dtype=int)
    else:
        raise ValueError('Invalid zoom value. Must be under 21.')
        return

    

def pixelToWorld((pixel_x,pixel_y),z):
    scale = np.float(np.power(2,z))
    world_x = np.float(pixel_x) / scale
    world_y = np.float(pixel_y) / scale
    return (world_x,world_y)


def worldToLatLon((world_x,world_y)):
    lng = ((world_x/TILE_SIZE) - 0.5)*360
    p = np.exp((-(world_y/TILE_SIZE)+.5)*np.pi*4)
    print p
    print ((1-p)/(1+p))
    lat = np.float(180/np.pi)*(np.arcsin((p-1)/(1+p)))
    return ((lat,lng))


