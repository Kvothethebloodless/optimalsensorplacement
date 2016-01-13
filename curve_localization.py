import numpy as np

import matplotlib.pyplot as plt

import arena as arena

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

print sum
distancevector = np.array(buff)

max_distance = np.max(distance_vector)


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
