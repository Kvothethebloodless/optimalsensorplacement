import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# import matplotlib.pyplot as plt
import gdsolver as gds
import psosolver as psos
import statelogger as stlog

from mayavi import mlab

global no_sensors
global ndim

no_sensors = 4
ndim = 2


def create_sensorarray(n_dim,n_sensors):
    return np.random.random((n_sensors,n_dim))*10


def plot_surface(fitfunc):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X1 = np.arange(-50, 50, 0.25)
    Y2 = np.arange(-50, 50, 0.25)
    X, Y = np.meshgrid(X1, Y2)
    Z = np.empty((np.size(X1), np.size(Y2)))
    for i in range(np.size(X1)):
        for j in range(np.size(Y2)):
            Z[i, j] = fitfunc(np.array([X1[i], Y2[j]]))

    Z = np.array(Z)
    # R = np.sqrt(X**2 + Y**2)
    # Z = np.sin(R)
    s = mlab.mesh(X, Y, Z);
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # ax.set_zlim(-1.01, 1.01)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


class localize():
    def __init__(self, no_sensors,sensor_loc):
        self.no_sensors = no_sensors
        self.sensor_loc = sensor_loc;
        self.create_target_object()
        self.get_original_ranges()
        self.get_noisy_ranges()

    def dist_from_ithsensor(self, i, point):
        if i > self.no_sensors:
            print "Exceeded"
            return
        else:
            return np.linalg.norm(point - self.sensor_loc[i])

    def gradient_score(self, point):
        point = np.array(point);
        dim = point.shape[0];
        gradi = np.empty(dim)
        dist_vector = [self.dist_from_ithsensor(i, point) for i in range(self.no_sensors)]
        dist_vector = np.array(dist_vector)
        common_factor_vector = [1 - ((self.noisy_ranges[i]) / dist_vector[i]) for i in range(self.no_sensors)]
        common_factor_vector = np.array(common_factor_vector)
        dim_diff_vector = point - self.sensor_loc;
        dim_gradient_vector = np.transpose(common_factor_vector * np.transpose(dim_diff_vector));
        dim_gradient = np.sum(dim_gradient_vector, axis=0)
        return dim_gradient * (2. / self.no_sensors)

    # grad_presum_vector = [np.dot(common_factor_vector[i],dim_diff_vector[i])

    # for i in range(	dim):
    ##gradi[dim] = 2*(self.noisy_ranges[i]/self.dist_from_ithsensor(i,)


    # self.sensor_loc =np.array([[1,2],[3,4],[5,6]])
    #

    def create_target_object(self):
        # self.target_loc = np.random.random((1,2))*10
        self.target_loc = [5, 5]

    def get_original_ranges(self):
        self.orig_ranges = self.sensor_loc - self.target_loc;
        self.orig_ranges = np.linalg.norm(self.orig_ranges, axis=1)

    def get_noisy_ranges(self):
        sigma = .1;

        mean_vector = self.orig_ranges;
        path_loss_coeff = 0.01;
        variance_vector = (sigma) * (np.power(self.orig_ranges, path_loss_coeff));
        # print mean_ve                                                                                                                                                                                                                                                   ctor
        # print variance_vector
        self.mean = mean_vector
        self.var = variance_vector
        nse = np.empty(self.no_sensors)
        for i in range(self.no_sensors):
            nse[i] = np.random.normal(mean_vector[i], variance_vector[i])
            print nse[i]
        # nse = np.array(n)

        self.noisy_ranges = nse

    def get_score(self, particle_loc):
        score = 0;
        cartesian_distance = np.linalg.norm(particle_loc - self.sensor_loc, axis=1)
        # print cartesian_distance
        # cartesian_distance = np.power(cartesian_distance,.5)
        # print cartesian_distance
        score_vector = self.noisy_ranges - cartesian_distance;
        # print score_vector
        score = np.mean(np.power(score_vector, 2))

        return score

    # def initalize_swarm_param(self):


loc_algo = localize(no_sensors,create_sensorarray(ndim,no_sensors));
#loc_algo = localize(no_sensors,np.load('psosolsens.npy'));
#loc_algo = localize(no_sensors,np.load('gdssolsens.npy'));
#loc_algo = localize(no_sensors,np.load('fr.npy'));

# indx = 0;
# lst_score = float("inf");
# pos_tracker = np.empty(np.shape(pso.current_pos));
# spread = 0;
# centroid = np.empty(pso.no_dim);

sensor_arena = loc_algo

point = np.random.random((1, ndim))
iterindx = 1
gdsolver = gds.GDS(sensor_arena.get_score, sensor_arena.gradient_score, point)
psosolver = psos.PSO(loc_algo.get_score, 10, ndim, 1, 1.5, 1, point);

psodata = stlog.statelogger('psodata2','psolog', np.vstack((psosolver.current_pos,sensor_arena.sensor_loc,sensor_arena.target_loc)), psosolver.centroid, psosolver.spread,psosolver.globalmin) #Plotter is  just expecting these.

# gddata = stlog.statelogger('gddata2', 'gdslog',np.vstack((gdsolver.new_point, sensor_arena.sensor_loc, sensor_arena.target_loc)),
#                            gdsolver.new_point, 0,gdsolver.sensor_arena.get_score(gdsolver.new_point)) #Combined position data of moving point sensors and sensor target, i.e the object to be localized, in the first vstack array.
#Dummy gdsolver.new_point to replace centroid that the plotter is expecting in the second term.
indx = 0;
# while np.abs(np.amin(pso.curr_score)-lst_score)>.001:
while indx < 300:
    psosolver.update_pos();
    psosolver.update_currscores();
    psosolver.update_selfmin();
    psosolver.update_globalmin();
    psosolver.update_velocities();
    (centroid, spread) = psosolver.calc_swarm_props();
    print('\n \n \n \n \n')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n')
    print("The running index is " + str(indx) + "\n")
    print(str(psosolver.globalminlocation))
    indx = indx + 1;
    if psosolver.no_particles < 20:
        psosolver.report();


    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n')
    curr_score = sensor_arena.get_score(point);
    point = gdsolver.get_descented_point(point);
    descented_score = sensor_arena.get_score(point);
    deltascore = descented_score - curr_score;
    curr_score = descented_score;
    gdsolver.report();
    psodata.add_state( np.vstack((psosolver.current_pos,sensor_arena.sensor_loc,sensor_arena.target_loc)),psosolver.centroid,psosolver.spread,psosolver.globalmin)
    # gddata.add_state(np.vstack((gdsolver.new_point, sensor_arena.sensor_loc, sensor_arena.target_loc)),
    #                  gdsolver.new_point, 0,gdsolver.sensor_arena.get_score(gdsolver.point))
#
##pso.plot_score()

# gdsolver.plot_score()
# plt.show()

# plt.plot(np.load('psodata.npy')[3], 'r')
# plt.show()
# plt.plot(np.load('gddata.npy')[3], 'b')
# plt.show()
