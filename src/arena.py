import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import axes3d, Axes3D 
import matplotlib.pyplot as plt


# global no_sensors
# global ndim

#Creates a physical array of sensors and target. And has methods to calculate some quantities like,  distances, scores, gradient of the score at a point in the arena, score of the point, 3d plot of the surface of the score.


class arena(): 
    """This class implements an arena. An arena, in compliance with regular meaning, space where the sensors and target are placed. It also takes care of interactions between the sensors and target, like, sampling noisy distance measurements, calculating score of position based on the distance values, etc.


      Args:
          no_sensors (int) : No of sensors measuring the target's distance.

          ndim (int) : No of dimensions of the space that the arena exists. 2 to 3 dimensions permissible.

          sensor_loc (numpy ndarray): Of size (n,ndim) contains all the sensors' locations in the arena. Necessary argument.

          target_loc (numpy ndarray): Of size (1,ndim). Denotes the target location. Necessary argument.
          """




    def __init__(self, no_sensors, ndim, sensor_loc, target_loc):
        self.no_sensors = no_sensors
        self.sensor_loc = sensor_loc
       # self.create_target_object()
        self.target_loc = target_loc
        self.get_original_ranges()
        self.get_noisy_ranges()
        self.target_loc = target_loc
        self.ndim = ndim

    def dist_from_ithsensor(self, i, point):
        """ 
        Gives distances from ith sensor to the point.
        Returns:
            distance (int):
        Raises:
            Prints exceeded if asked for a sensor's distance measurement which is not availale in
            the array.
        """
        if i > self.no_sensors:
            print "Exceeded"
            return
        else:
            return np.linalg.norm(point - self.sensor_loc[i])
    
    def gradient_score(self, point):
        point = np.array(point)
        dim = point.shape[0]
        gradi = np.empty(dim)
        dist_vector = [self.dist_from_ithsensor(i, point) for i in range(self.no_sensors)]
        dist_vector = np.array(dist_vector)
        common_factor_vector = [1 - ((self.noisy_ranges[i]) / dist_vector[i]) for i in range(self.no_sensors)]
        common_factor_vector = np.array(common_factor_vector)
        dim_diff_vector = point - self.sensor_loc
        dim_gradient_vector = np.transpose(common_factor_vector * np.transpose(dim_diff_vector))
        dim_gradient = np.sum(dim_gradient_vector, axis=0)
        return dim_gradient * (2. / self.no_sensors)

    # grad_presum_vector = [np.dot(common_factor_vector[i],dim_diff_vector[i])

    # for i in range(	dim):
    ##gradi[dim] = 2*(self.noisy_ranges[i]/self.dist_from_ithsensor(i,)


    # self.sensor_loc =np.array([[1,2],[3,4],[5,6]])
    #

        self.target_loc = [5, 5]

    def get_original_ranges(self):
        a = self.sensor_loc - self.target_loc
        self.orig_ranges = np.linalg.norm(a, axis=1)

    def get_noisy_ranges(self):
        sigma = .1

        mean_vector = self.orig_ranges
        path_loss_coeff = 2.77 
        variance_vector = sigma * (np.power(self.orig_ranges, path_loss_coeff))
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
        score = 0
        print(particle_loc)
        a = particle_loc - self.sensor_loc
        print(a)
        cartesian_distance = np.linalg.norm(a, axis=1)
        print(cartesian_distance)
        # print cartesian_distance
        # cartesian_distance = np.power(cartesian_distance,.5)
        # print cartesian_distance
        score_vector = self.noisy_ranges - cartesian_distance
        print(score_vector)
        # print score_vector
        score = np.mean(np.power(score_vector, 2))

        return score
        # def initalize_swarm_param(self):

    def plotscoresurface(self):
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator, FormatStrFormatter
        #import matplotlib.pyplot as plt
        #import numpy as np

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        X = np.arange(-5, 5, 0.25)
        Y = np.arange(-5, 5, 0.25)
        X, Y = np.meshgrid(X, Y)
        Z = self.get_score((X, Y))
        (X, Y)
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                                                      linewidth=0, antialiased=False)
        #ax.set_zlim(-1.01, 1.01)

        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()
    def plot_surface(self):
        fig = plt.figure()
       # ax = fig.gca(projection='3d')
        ax = Axes3D(fig)
        X1 = np.arange(-5, 5, 0.25)
        Y2 = np.arange(-5, 5, 0.25)
        X, Y = np.meshgrid(X1, Y2)
        Z = np.empty((np.size(X1), np.size(Y2)))
        for i in range(np.size(X1)):
            for j in range(np.size(Y2)):
                Z[i, j] = self.get_score(np.array([X1[i], Y2[j]]))

        Z = np.array(Z)
        # R = np.sqrt(X**2 + Y**2)
        # Z = np.sin(R)
#        s = mlab.mesh(X, Y, Z)
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
        # ax.set_zlim(-1.01, 1.01)

        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()



