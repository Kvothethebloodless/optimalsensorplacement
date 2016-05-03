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
from src import terrain
from src import visualization


def scorefunc(dw):
	points = np.array(r.dwtoxyz(dw))
	true_points = points.T
	return ar.get_score(true_points)




(terrain_x,terrain_y,terrain_z) = np.load('simulationdata/sub_terrain.npy')
(road_xcords,road_ycords) = np.load('simulationdata/visualization/road_cords.npy')
(road_x,road_y) = (terrain_x[road_ycords,road_xcords],terrain_y[road_ycords,road_xcords])
r = terrain.road(road_xcords,road_ycords,(terrain_x,terrain_y,terrain_z))
print (r.maxdistance)
r.plotroad()



sensor_loc = [[39,82],[148,158],[214,57]]
sensor_real_loc = [[r.x[ele],r.y[ele],r.z[ele]]  for ele in sensor_loc]

target_dw = [500,2]
target_real_loc = r.dwtoxyz(target_dw)
ar = arena.arena(3,3,sensor_real_loc,target_real_loc)

ssb = np.array([10,-2],[r.maxdistance-10,2])

pslr = psos.PSO(scorefunc,2,searchspaceboundaries=ssb)
pslr.solve_iers(100,verbose=True,out=True)