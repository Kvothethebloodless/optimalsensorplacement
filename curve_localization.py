import numpy as np
import matplotlib.pyplot as plt

#The polynomial is - [1.89427432, 1.8132651, -0.44164664, 0.03026042]


#Generate points in x and y co-ordinates with equal cartesian distances by interpolating.
# Its hard to do that, so I am jsut generating points with equal distance on the x-axis. If I generate enough number of points this way, I guess, the distance wont't
#matter that much.


# x is in range 0 to 10;

road_curve_polynomial = np.poly1d([1.89427432, 1.8132651, -0.44164664, 0.03026042])


no_points = 10000;
roadbegin_x = 0;
roadend_x = 10;
road_xpoints = np.linspace(roadbegin_x,roadend_x,no_points);
road_ypoints = road_curve(road_xpoints);
road_points = zip(road_xpoints,road_ypoints);


plt.ion()
plt.figure()
plt.plot(road_ypoints,road_xpoints);
plt.title('Road Shape');
plt.xlablel('in mts');
plt.ylabel('in mts');
plt.show()


distance_vector = np.linalg.norm(road_points-np.roll(road_points,-2),axis=1)
distance_vector[0] = 0;

max_distance = np.max(distance_vector);

def scorefunc(dist):
    dist()



