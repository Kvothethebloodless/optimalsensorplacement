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
	# pdb.set_trace()
	points = np.array(r.dwtoxyz(dw[0],dw[1]))
	true_points = points.T
	return ar.get_score(true_points)


# pdb.set_trace()

(terrain_x,terrain_y,terrain_z) = np.load('simulationdata/visualize/subterrain.npy')
(road_xcords,road_ycords) = np.load('simulationdata/visualize/road_cords.npy')
(road_x,road_y) = (terrain_x[road_ycords,road_xcords],terrain_y[road_ycords,road_xcords])
r = terrain.road((road_xcords,road_ycords),(terrain_x,terrain_y,terrain_z))
print (r.maxdistance)
# r.plotroad()



sensor_loc = [[39,82],[148,158],[214,57]]
sensor_real_loc = np.array([[r.x[ele[0],ele[1]],r.y[ele[0],ele[1]],r.z[ele[0],ele[1]]]  for ele in sensor_loc])

target_dw = [100,0]
target_real_loc = r.dwtoxyz(target_dw[0],target_dw[1])
ar = arena.arena(3,3,sensor_real_loc,target_real_loc)

ssb = np.array([[10,-3],[r.maxdistance-10,3]])

pslr = psos.PSO(scorefunc,2,searchspaceboundaries=ssb)
# pslr.solve_iters(10,verbose=True,out=True)

sol = []
for i in range(10):
	sol.append(pslr.solve_convergence(0.001,checkiters=20,out=True))

