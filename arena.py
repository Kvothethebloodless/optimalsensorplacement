import numpy as np


# global no_sensors
# global ndim



class arena:  # Class to hold the physical space and make some calculations on it.
    def __init__(self, no_sensors, ndim, sensor_loc, target_loc):
        self.no_sensors = no_sensors
        self.sensor_loc = sensor_loc
        self.create_target_object()
        self.get_original_ranges()
        self.get_noisy_ranges()
        self.target_loc = target_loc
        self.ndim = ndim

    def dist_from_ithsensor(self, i, point):
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

    def create_target_object(self):
        # self.target_loc = np.random.random((1,2))*10
        self.target_loc = [5, 5]

    def get_original_ranges(self):
        self.orig_ranges = self.sensor_loc - self.target_loc
        self.orig_ranges = np.linalg.norm(self.orig_ranges, axis=1)

    def get_noisy_ranges(self):
        sigma = .1

        mean_vector = self.orig_ranges
        path_loss_coeff = 0.01
        variance_vector = sigma * (np.power(self.orig_ranges, path_loss_coeff))
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
        score = 0
        cartesian_distance = np.linalg.norm(particle_loc - self.sensor_loc, axis=1)
        # print cartesian_distance
        # cartesian_distance = np.power(cartesian_distance,.5)
        # print cartesian_distance
        score_vector = self.noisy_ranges - cartesian_distance
        # print score_vector
        score = np.mean(np.power(score_vector, 2))

        return score
        # def initalize_swarm_param(self):
