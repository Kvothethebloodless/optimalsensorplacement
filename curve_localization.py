import numpy as np

import matplotlib.pyplot as plt

import arena as arena
import psosolver as psos  # Pso solver
import statelogger as stlog  # Logging states and state variables

ndim = 1;
# The polynomial is - [1.89427432, 1.8132651, -0.44164664, 0.03026042]


# Generate points in x and y co-ordinates with equal cartesian distances by interpolating.
# Its hard to do that, so I am jsut generating points with equal distance on the x-axis. If I generate enough number of points this way, I guess, the distance wont't
# matter that much.


# x is in range 0 to 10;
coeffarray = [1.89427432, 1.8132651, -0.44164664, 0.03026042];


def create_sensorlocarray(n_dim, n_sensors):
	return np.random.random((n_sensors, n_dim)) * 10


road_curve_polynomial = np.poly1d(coeffarray[::-1])

no_points = 100000
roadbegin_x = 0
roadend_x = 10
road_xpoints = np.linspace(roadbegin_x, roadend_x, no_points)
road_ypoints = road_curve_polynomial(road_xpoints)

road_points = zip(road_xpoints, road_ypoints)
road_points = np.array(road_points)

plt.ion()
plt.figure()
# plt.plot( (((road_xpoints/no_points)*(roadend_x-roadbegin_x))+roadbegin_x),road_ypoints)
plt.plot(road_xpoints, road_ypoints);
plt.title('Road Shape')
plt.xlabel('in mts')
plt.ylabel('in mts')
plt.show()

distance_vector = np.linalg.norm(road_points - np.roll(road_points, -2), axis=1)
distance_vector = np.roll(distance_vector, 1)
distance_vector[0] = 0


def create_target():
	targetdist = 5
	return getpoint(5, distancevector, road_points)


print ('The resolution at maximum is ', np.max(distance_vector));

buff = []
sum = 0
for i in range(no_points):
	sum += distance_vector[i];
	buff.append(sum)


distancevector = np.array(buff)

max_distance = np.max(distancevector)
roadlength = max_distance;


def nearest_dist(ele, vector):
	if (np.sum(vector == ele)):
		return ele

	else:
		less_near = np.max(
				vector[vector < ele])  # To find the nearest smaller number which is the maximum of lesser elements.
		greater_near = np.min(vector[vector > ele])
		dist_lesser = np.abs(ele - less_near)
		dist_greater = np.abs(ele - greater_near)
		if dist_lesser > dist_greater:
			return greater_near
		else:
			return less_near


def getpointfromdistance(dist, dist_array, points_array):
	if np.sum(dist_array == dist):
		return points_array[dist_array == dist]
	else:
		print('Element not found. Breaking...')
		exit()


def getpoint(dist, dist_array, points_array):
	nearest_dis = nearest_dist(dist, dist_array);
	nearest_point = getpointfromdistance(nearest_dis, dist_array, points_array)
	return nearest_point


sensorloc = create_sensorlocarray(2, 3)
targetloc = create_target()
curr_arena = arena.arena(3, 2, sensorloc, targetloc);


def scorefunc(dist):
	point = getpoint(dist, distancevector, road_points)
	score = curr_arena.get_score(point)
	return score


# def gradscorefunc(dist):
# 	point = getpoint(dist, distancevector, road_points)
# 	dist_left = (np.roll(road_points==point,1))
# 	pointleft = road_points(np.roll(road_points==point,-1))

def convertpsotopos(psoobject):
	l = []
	for pos in psoobject.current_pos:
		l.append(np.array(getpoint(pos, distancevector, road_points)))
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
	psoslr = psos.PSO(curr_arena.get_score, 10, ndim, 1, 1.5, 1, ExampleSolSpacePoint,
					  [0 * np.ones(ndim), roadlength * np.ones(ndim)])
	psoarray = convertpsotopos(psoslr)

	psodata = stlog.statelogger('psodata2', 'psolog',
								np.vstack((psoarray, curr_arena.sensor_loc, curr_arena.target_loc)),
								psoslr.centroid, psoslr.spread,
								psoslr.globalmin)  # Plotter is  just expecting these.

	# gddata = stlog.statelogger('gddata2', 'gdslog',np.vstack((gdsolver.new_point, curr_arena.sensor_loc, curr_arena.target_loc)),
	#                            gdsolver.new_point, 0,gdsolver.curr_arena.get_score(gdsolver.new_point)) #Combined position data of moving point sensors and sensor target, i.e the object to be localized, in the first vstack array.
	# Dummy gdsolver.new_point to replace centroid that the plotter is expecting in the second term.
	indx = 0
	# while np.abs(np.amin(pso.curr_score)-lst_score)>.001:
	point = ExampleSolSpacePoint

	while indx < 300:
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
