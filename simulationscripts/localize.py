import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# import matplotlib.pyplot as plt
import solvers.gdsolver as gds  # GDS solver
import solvers.psosolver as psos  # Pso solver
import utilities.statelogger as stlog  # Logging states and state variables

import src.arena as ar  # Class to place sensors and target in an imaginary arena and calculate score, gradient_score, ranges,noisy ranges etc.

global no_sensors
global ndim

no_sensors = 4
ndim = 2


def create_sensorlocarray(n_dim, n_sensors):
    return np.random.random((n_sensors, n_dim)) * 10


def create_target(ndim):
    return 5 * np.ones(ndim)


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
    s = mlab.mesh(X, Y, Z)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # ax.set_zlim(-1.01, 1.01)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


def solveforlocation():
    sensorloc_array = create_sensorlocarray(ndim, no_sensors)
    target_loc = create_target(ndim)
    curr_arena = ar.arena(no_sensors, ndim, sensorloc_array,target_loc)

    # curr_arena = localize(no_sensors,np.load('psosolsens.npy'));
    # curr_arena = localize(no_sensors,np.load('gdssolsens.npy'));
    # curr_arena = localize(no_sensors,np.load('fr.npy'));

    # indx = 0;
    # lst_score = float("inf");
    # pos_tracker = np.empty(np.shape(pso.current_pos));
    # spread = 0;
    # centroid = np.empty(pso.no_dim);


    ExampleSolSpacePoint = np.random.random((1, ndim))
    iterindx = 1
    gdsolver = gds.GDS(curr_arena.get_score, curr_arena.gradient_score, ExampleSolSpacePoint)
    psosolver = psos.PSO(curr_arena.get_score, 10, ndim, 1, 1.5, 1, ExampleSolSpacePoint,
                         [-10 * np.ones(ndim), 10 * np.ones(ndim)])

    psodata = stlog.statelogger('localize_psodata', 'localize_psolog',
                                np.vstack((psosolver.current_pos, curr_arena.sensor_loc, curr_arena.target_loc)),
                                psosolver.centroid, psosolver.spread,
                                psosolver.globalmin)  # Plotter is  just expecting these.

    # gddata = stlog.statelogger('gddata2', 'gdslog',np.vstack((gdsolver.new_point, curr_arena.sensor_loc, curr_arena.target_loc)),
    #                            gdsolver.new_point, 0,gdsolver.curr_arena.get_score(gdsolver.new_point)) #Combined position data of moving point sensors and sensor target, i.e the object to be localized, in the first vstack array.
    # Dummy gdsolver.new_point to replace centroid that the plotter is expecting in the second term.
    indx = 0
    # while np.abs(np.amin(pso.curr_score)-lst_score)>.001:
    point = ExampleSolSpacePoint

    while indx < 300:
        psosolver.update_pos()
        psosolver.update_currscores()
        psosolver.update_selfmin()
        psosolver.update_globalmin()
        psosolver.update_velocities()
        (centroid, spread) = psosolver.calc_swarm_props()
        print('\n \n \n \n \n')
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n')
        print("The running index is " + str(indx) + "\n")
        print(str(psosolver.globalminlocation))
        indx += 1
        if psosolver.no_particles < 20:
            psosolver.report()

        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n')
        curr_score = curr_arena.get_score(point)
        point = gdsolver.get_descented_point(point)
        descented_score = curr_arena.get_score(point)
        deltascore = descented_score - curr_score
        curr_score = descented_score
        gdsolver.report()
        psodata.add_state(np.vstack((psosolver.current_pos, curr_arena.sensor_loc, curr_arena.target_loc)),
                          psosolver.centroid, psosolver.spread, psosolver.globalmin)
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


if __name__ == "__main__":
    solveforlocation()
