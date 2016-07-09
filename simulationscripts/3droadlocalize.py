from __future__ import division

import numpy as np
from scipy import interpolate
from scipy.integrate import quad
from scipy.optimize import fsolve as fs

import matplotlib.pyplot as plt

(terrain_x,terrain_y,terrain_z) = np.load('simulationdata/corrected_terrain_mt.npy')



terrainalt_spline = interpolate.RectBivariateSpline(terrain_x[0],terrain_y[:,0],terrain_z)
getz = terrainalt_spline.ev

def create_spline_curve():
    x = terrain_x[0,[0,200,400,600,1000]]
    y = terrain_y[[230,400,600,400,300],0]
	tck = interpolate.splrep(x, y, s=0)
    return tck

def getlengthfromspline(x,tck,length):
    I = quad(lambda x:np.sqrt(1+(np.power(interpolate.splev(x,tck,der=1),2))),0,x)[0]
    return I-length

tck = create_spline_curve()
fullroad_x = terrain_x[0]
fullroad_y = interpolate.splev(fullroad_x,tck,der=0)
fullroad_z = []

for i in range(np.size(fullroad_x)):
	fullroad_z.append(getz(fullroad_x[i],fullroad_y[i]))

fullroad_z = np.array(fullroad_z)
road_length = getlengthfromspline(terrain_x[0][-1],tck,0)


def getpoint_spline(dist,tck):
    x = fs(getlengthfromspline,0.0,(tck,dist))[0] #fs is a solver. It solves for dist_spline  =
    #req_dist
    return x,interpolate.splev(x,tck,der=0).item()






def create_sensorlocarray(n_dim, n_sensors):
	return np.random.random((n_sensors, n_dim)) * 10




def create_target():
	targetdist = 5
	return getpoint_spline(targetdist,tck)


def createwidthroad(width,tck):
	terrainwithwideroad = np.copy(terrain_y)

	for i in range(1024):
		y_col = terrain_y[:,i]
		y_road = fullroad_y[i]
		costheta = 1/np.sqrt(1+(np.power(interpolate.splev(fullroad_x[i],tck,der=1),2)))
		mask = np.abs(y_col-y_road) < ((width/2)*costheta)
		y_colnew = np.logical_and(np.ones_like(y_col), mask)
		terrainwithwideroad[:,i] = y_colnew

	plt.imshow(terrainwithwideroad)
	plt.show()


def convertdwtoxy(tupletdw):
	[d,w] = tupletdw
	(x_point,y_point) = getpoint_spline(d,tck)
	fprime = (interpolate.splev(x_point,tck,der=1))
	costheta = 1/np.sqrt(1+(np.power(fprime,2)))
	new_x = x_point- (((fprime)*w)*costheta)
	new_y = y_point + (w*costheta)
	return np.array([new_x,new_y])

def convertdwtoxyarray(array_dw):
	new = []
	for ele in array_dw:
		new.append(convertdwtoxy([ele[0],ele[1]]))
	return np.array(new)


def convertdwtoxyz(tupletdw):
	[d,w] = tupletdw
	(x_point,y_point) = getpoint_spline(d,tck)
	fprime = (interpolate.splev(x_point,tck,der=1))
	costheta = 1/np.sqrt(1+(np.power(fprime,2)))
	new_x = x_point- (((fprime)*w)*costheta)
	new_y = y_point + (w*costheta)
	new_z = terrainalt_spline.ev(new_x,new_y)
	return np.array([new_x,new_y,new_z])


def convertdwtoxyzarray(array_dw):
	new = []
	for ele in array_dw:
		new.append(convertdwtoxyz([ele[0],ele[1]]))
	return np.array(new)

#a = 	createwidthroad(40,tck)

randa = np.array([[565, 711],[797, 724], [191, 593]])
sensorloc1 = np.array([terrain_x[565, 711],terrain_y[565, 711],terrain_z[565, 711]])
sensorloc2 = np.array([terrain_x[797, 724],terrain_y[797, 724],terrain_z[797, 724]])
sensorloc3 = np.array([terrain_x[191, 593],terrain_y[191, 593],terrain_z[191, 593]])
sensorloc = np.array([sensorloc1,sensorloc2,sensorloc3])



#sensorloc = create_sensorlocarray(2, 3)
# targetloc_xy = convertdwtoxy([500,20])
# targetloc_z = terrainalt_spline.ev(targetloc_xy[0],targetloc_xy[1])
# targetloc = np.array([targetloc_xy[0],targetloc_xy[1],targetloc_z])

targetloc = convertdwtoxyz([500,30])



curr_arena = arena.arena(3, 3, sensorloc, targetloc);
#
# road_xpoints = np.linspace(0,10,1000)
# road_ypoints = interpolate.splev(road_xpoints,tck,der=0)
# plt.plot(road_xpoints,road_ypoints)
# plt.ion()
# plt.show()




def scorefunc_spline(dwarray):
    point = convertdwtoxyz(dwarray)
    score = curr_arena.get_score(point)
    return score
# def gradscorefunc(dist):
# 	point = getpoint(dist, distancevector, road_points)
# 	dist_left = (np.roll(road_points==point,1))
# 	pointleft = road_points(np.roll(road_points==point,-1))

def convertpsotopos(psoobject):
	# l = []
	# for pos in psoobject.current_pos:
	# 	l.append(np.array(getpoint_spline(pos,tck)))
	# print (np.shape(np.array(l)))
	# return np.vstack(l)
	return convertdwtoxyzarray(psoobject.current_pos)

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


	ExampleSolSpacePoint = np.array([1500,10])
	iterindx = 1
	# gdsolver = gds.GDS(scorefunc, curr_arena.gradient_score, ExampleSolSpacePoint)
	# psoslr = psos.PSO(scorefunc_spline, 10, 2, 1.5, 4, 2, ExampleSolSpacePoint,
#					  np.array([np.array([100,0]),np.array([8000,40])]))
	# psoarray = convertpsotopos(psoslr)

	# psodata = stlog.statelogger('splineroad3dlocalize_psodata2', 'splineroad3dlocalize_psolog',
	# 							np.vstack((psoarray, curr_arena.sensor_loc, curr_arena.target_loc)),
	# 							psoslr.centroid, psoslr.spread,
	# 							psoslr.globalmin)  # Plotter is  just expecting these.

	# gddata = stlog.statelogger('gddata2', 'gdslog',np.vstack((gdsolver.new_point, curr_arena.sensor_loc, curr_arena.target_loc)),
	#                            gdsolver.new_point, 0,gdsolver.curr_arena.get_score(gdsolver.new_point)) #Combined position data of moving point sensors and sensor target, i.e the object to be localized, in the first vstack array.
	# Dummy gdsolver.new_point to replace centroid that the plotter is expecting in the second term.
	indx = 0
	# while np.abs(np.amin(pso.curr_score)-lst_score)>.001:
	# point = ExampleSolSpacePoint



	# while indx < 50:
	# 	psoslr.update_pos()
	# 	psoslr.update_currscores()
	# 	psoslr.update_selfmin()
	# 	psoslr.update_globalmin()
	# 	psoslr.update_velocities()
	# 	(centroid, spread) = psoslr.calc_swarm_props()
	# 	print('\n \n \n \n \n')
	# 	print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n')
	# 	print("The running index is " + str(indx) + "\n")
	# 	print(str(psoslr.globalminlocation))
	# 	indx += 1
	# 	if psoslr.no_particles < 20:
	# 		psoslr.report()
	#
	# 	print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n')
	# 	# curr_score = curr_arena.get_score(point)
	# 	# point = gdsolver.get_descented_point(point)
	# 	# descented_score = curr_arena.get_score(point)
	# 	# deltascore = descented_score - curr_score
	# 	# curr_score = descented_score
	# 	# gdsolver.report()
	# 	psoarray = convertpsotopos(psoslr)
	# 	psodata.add_state(np.vstack((psoarray, curr_arena.sensor_loc, curr_arena.target_loc)),
	# 					  psoslr.centroid, psoslr.spread, psoslr.globalmin)
	# 	globalmin = psoslr.globalminlocation


	# solveforlocation()
ssb =  np.array([np.array([100,0]),np.array([8000,40])])
psoslr = psos.PSO(scorefunc_spline,2,searchspaceboundaries=ssb,maxvel=100)
soltion = psoslr.solve_iters(100,out=True,verbose=True)
	# finalpoint = convertdwtoxyz(globalmin)
	# print ('The location estimate is: ' + str(finalpoint))

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


#np.save(os.path.join('simulationdata/','spline_road'), [road_xpoints, road_ypoints])

