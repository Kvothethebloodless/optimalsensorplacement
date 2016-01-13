import numpy as np
import math
import matplotlib.pyplot as plt


class PSO():
    def __init__(self, no_particles, no_dim, self_accel_coeff, global_accel_coeff, dt):
        """no_particles, no_dim, self_accel_coeff, global_accel_coeff, dt , A total of 5 parameters
		no_particles = Number of swarm particles to create and engage
		no_dim = Dimension of the search space 	"""

        self.no_particles = no_particles;
        self.no_dim = no_dim;

        self.self_accel_coeff = self_accel_coeff;
        self.global_accel_coeff = global_accel_coeff;

        self.velocity = np.random.random((self.no_particles, self.no_dim));

        self.current_pos = np.random.random((self.no_particles, self.no_dim)) * 100
        self.paramspacebounds = [-10, 10];

        self.dt_velocity = 1;
        self.dt_pos = 1
        self.weight = .35
        self.maxvel = 5;
        self.minvel = -5;
        self.selfminlocation = np.random.random((self.no_particles, self.no_dim));
        self.selfminval = self.funcdef(self.selfminlocation, self.no_particles);

        self.globalmin = np.min(self.selfminval);
        self.globalminlocation = self.selfminlocation[np.argmin(self.selfminval)];

        self.curr_score = self.funcdef(self.current_pos, self.no_particles);
        self.posseries_data = []  # np.empty(np.shape(self.current_pos))
        self.centroidseries_data = []  # np.empty(self.no_dim);
        self.spreadseries_data = [];
        self.centroid = np.empty(self.no_dim)
        self.spread = 0;

    def update_selfmin(self):
        update_req_array = self.curr_score < self.selfminval

        self.selfminval = np.multiply(update_req_array, self.curr_score) + np.multiply(np.invert(update_req_array),
                                                                                       self.selfminval);

        update_req_array = np.transpose(update_req_array);

        update_req_array = np.reshape(update_req_array, (self.no_particles, 1))
        print(np.shape(update_req_array));

        print(np.shape(self.current_pos));
        self.selfminlocation = (update_req_array * self.current_pos) + (
        np.invert(update_req_array) * self.selfminlocation);

    def update_globalmin(self):
        curr_globalmin = np.amin(self.curr_score);
        state = curr_globalmin < self.globalmin
        state = np.squeeze(state)
        self.globalmin = np.multiply(state, curr_globalmin) + np.multiply(np.invert(state), self.globalmin);
        self.globalminlocation = (state * curr_globalmin) + (np.invert(state) * self.globalminlocation);

    def funcdef(self, locs, no_particles):
        score_curr = np.empty(no_particles);
        for i in range(no_particles):
            score_curr[i] = np.linalg.norm(locs[i])
            print(score_curr[i])
        return score_curr

    def update_currscores(self):
        self.curr_score = self.funcdef(self.current_pos, self.no_particles);

    def update_velocities(self):
        self_accel = np.random.random() * self.selfminlocation - self.current_pos;
        global_accel = np.random.random() * self.globalminlocation - self.current_pos;
        accel = self.self_accel_coeff * self_accel + self.global_accel_coeff * global_accel;
        self.velocity = self.weight * self.velocity + accel * self.dt_velocity;
        self.velocity[self.velocity > self.maxvel] = 5;
        self.velocity[self.velocity < self.minvel] = -5;

    def update_pos(self):
        self.current_pos = self.current_pos + self.velocity * self.dt_pos;

    def calc_swarm_props(self):
        centroid = np.mean(self.current_pos, axis=0);
        spread = 0;
        for i in range(self.no_particles):
            spread = np.linalg.norm(self.current_pos - centroid) + spread;
        self.centroid = centroid
        self.spread = spread
        self.best_score = np.min(self.curr_score);
        return (centroid, spread)

    def valueIO(self):

        self.posseries_data.append(self.current_pos)  # np.hstack((self.posseries_data,self.current_pos))
        self.spreadseries_data.append(self.spread)  # = np.append(self.spreadseries_data,self.spread)
        self.centroidseries_data.append(self.centroid)  # = np.vstack((self.centroidseries_data,self.centroid))
        np.save('data_tracker',
                [np.array(self.posseries_data), np.array(self.centroidseries_data), np.array(self.spreadseries_data)])

    def report(self):
        print("The centroid is at " + str(self.centroid) + "\n");
        print("The spread is " + str(self.spread) + "\n");
        print("The positions are " + str(self.current_pos) + "\n");
        # print("-----");
        print("The current score is  " + str(self.curr_score) + "\n");
        print("------")
        # print("The current velocities are "+ str(self.velocity)+"\n");
        # print("#$#$#$#$#$#$ \n")

        print("The current best score  is" + str(np.min(self.curr_score)) + "\n");


pso = PSO(10, 2, .5, .5, .5);
indx = 0;
lst_score = float("inf");
pos_tracker = np.empty(np.shape(pso.current_pos));
spread = 0;
centroid = np.empty(pso.no_dim);
# while np.abs(np.amin(pso.curr_score)-lst_score)>.001:
while indx < 100:
    pso.update_currscores();
    pso.update_selfmin();
    pso.update_globalmin();
    pso.update_velocities();
    pso.update_pos();
    (centroid, spread) = pso.calc_swarm_props();
    print('\n \n \n \n \n')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n')
    print("The running index is " + str(indx) + "\n")
    indx = indx + 1;
    pso.valueIO();
    pso.report();
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n')
