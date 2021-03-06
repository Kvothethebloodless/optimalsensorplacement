import numpy as np
import os
from scipy import interpolate
from scipy.optimize import fsolve as fs
from scipy.integrate import quad

import matplotlib.pyplot as plt

import src.arena as arena import solvers.psosolver as psos  # Pso solver
import utilities.statelogger as stlog  # Logging states and state variables


ndim = 1;
# The polynomial is - [1.89427432, 1.8132651, -0.44164664, 0.03026042]


# Generate points in x and y co-ordinates with equal cartesian distances by interpolating.
# Its hard to do that, so I am jsut generating points with equal distance on the x-axis. If I generate enough number of points this way, I guess, the distance wont't
# matter that much.




def create_spline_curve():
    x = [0,1,3,6,10]
    y = [0,4,7,3,17]
    tck = interpolate.splrep(x,y,s=0)
    return tck

def getlengthfromspline(x,tck,length):
    I = quad(lambda x:np.sqrt(1+(np.power(interpolate.splev(x,tck,der=1),2))),0,x)[0]
    return I-length

def getpoint_spline(dist,tck):
    x = fs(getlengthfromspline,0.0,(tck,dist))[0] #fs is a solver. It solves for dist_spline  =
    #req_dist
    return x,interpolate.splev(x,tck,der=0).item()


def create_sensorlocarray(n_dim, n_sensors):
	return np.random.random((n_sensors, n_dim)) * 10


tck = create_spline_curve()

def create_target():
	targetdist = 5
	return getpoint_spline(targetdist,tck)




roadlength = 21.4;


# targetloc = create_target()

sensorloc = create_sensorlocarray(2, 3)
targetloc = create_target()
curr_arena = arena.arena(3, 2, sensorloc, targetloc);

road_xpoints = np.linspace(0,10,1000)
road_ypoints = interpolate.splev(road_xpoints,tck,der=0)
plt.plot(road_xpoints,road_ypoints)
plt.ion()
plt.show()


def scorefunc_spline(dist):
    point =getpoint_spline(dist,tck)
    score = curr_arena.get_score(point)
    return score
# def gradscorefunc(dist):
# 	point = getpoint(dist, distancevector, road_points)
# 	dist_left = (np.roll(road_points==point,1))
# 	pointleft = road_points(np.roll(road_points==point,-1))

def convertpsotopos(psoobject):
	l = []
	for pos in psoobject.current_pos:
		l.append(np.array(getpoint_spline(pos,tck)))
	print (np.shape(np.array(l)))
	return np.vstack(l)


def solveforlocation():
	print ('Solving.... \n')
	# sensorloc_array = create_sensorlocarray(ndim, no_sensors)
	# target_loc = create_target(ndim)
	# curr_arena = ar.arena(no_sensors, ndim, sensorloc_array,rarget_loc)

	# curr_arena = localize(no_sensors,np.load('psosolsens.npy'));
	# curr_arena = localize(no_sensors,np.load('gdssolsens.npy'));
	# curr_arena = localize(no_sensors,np.load('fr.npy'));

	# indx = 0;
	# lst_score = float("inf");
	# pos_tracker = np.empty(np.shape(pso.current_pos));
	# spread = 0;
	# centroid = np.empty(pso.no_dim);


	ExampleSolSpacePoint = np.random.random(1) + 5;
	iterindx = 1
	# gdsolver = gds.GDS(scorefunc, curr_arena.gradient_score, ExampleSolSpacePoint)
	psoslr = psos.PSO(scorefunc_spline, 10, ndim, 1, 1.5, 1, ExampleSolSpacePoint,
					  [0 * np.ones(ndim), roadlength * np.ones(ndim)])
	psoarray = convertpsotopos(psoslr)

	psodata = stlog.statelogger('splinelocalize_psodata2', 'splinelocalize_psolog',
								np.vstack((psoarray, curr_arena.sensor_loc, curr_arena.target_loc)),
								psoslr.centroid, psoslr.spread,
								psoslr.globalmin)  # Plotter is  just expecting these.

	# gddata = stlog.statelogger('gddata2', 'gdslog',np.vstack((gdsolver.new_point, curr_arena.sensor_loc, curr_arena.target_loc)),
	#                            gdsolver.new_point, 0,gdsolver.curr_arena.get_score(gdsolver.new_point)) #Combined position data of moving point sensors and sensor target, i.e the object to be localized, in the first vstack array.
	# Dummy gdsolver.new_point to replace centroid that the plotter is expecting in the second term.
	indx = 0
	# while np.abs(np.amin(pso.curr_score)-lst_score)>.001:
	point = ExampleSolSpacePoint

	while indx < 1:
		psoslr.update_pos()
		psoslr.update_currscores()
		psoslr.update_selfmin()
		psoslr.update_globalmin()
		psoslr.update_velocities()
		(centroid, spread) = psoslr.calc_swarm_props()
		print('\n \n \n \n \n')
		print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n')
		print("The running index is " + str(indx) + "\n")
		print(str(psoslr.globalminlocation))
		indx += 1
		if psoslr.no_particles < 20:
			psoslr.report()

		print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n')
		# curr_score = curr_arena.get_score(point)
		# point = gdsolver.get_descented_point(point)
		# descented_score = curr_arena.get_score(point)
		# deltascore = descented_score - curr_score
		# curr_score = descented_score
		# gdsolver.report()
		psoarray = convertpsotopos(psoslr)
		psodata.add_state(np.vstack((psoarray, curr_arena.sensor_loc, curr_arena.target_loc)),
						  psoslr.centroid, psoslr.spread, psoslr.globalmin)
	# gddata.add_state(np.vstack((gdsolver.new_point, curr_arena.sensor_loc, curr_arena.target_loc)),
	#                  gdsolver.new_point, 0,gdsolver.curr_arena.get_score(gdsolver.point))
	#
	##pso.plot_score()

	# gdsolver.plot_score()
	# plt.show()

	# plt.plot(np.load('psodata.npy')[3], 'r')
	# plt.show()
	# plt.plot(np.load('gddata.npy')[3], 'b')
	# plt.show()


solveforlocation()
np.save(os.path.join('simulationdata/','spline_road'), [road_xpoints, road_ypoints])
