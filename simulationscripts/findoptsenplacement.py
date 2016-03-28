import numpy as np
import os
import src.optsenpmtscore as ops
import solvers.psosolver as pssol
import utilities.statelogger as stlg
import solvers.gdsolver as gdslr
import matplotlib.pyplot as plt

global no_sensors
global avgspace
global alpha
global sigma

global dim

dim = 2
no_sensors = 4
#avgspace = np.random.random((100,2))*10-(np.ones((100,2))*5)
road_points = np.load(os.path.join('simulationdata/','road.npy'))

avgspace = zip(road_points[0],road_points[1]) #For placement on a road #[5,5]+np.random.random((100,2))/10
avgspace_some = avgspace[0:100:10]
print ('Working on '+str(np.size(avgspace_some[0]))+'points')
alpha = 0.01
sigma = .1

def savesensorarray(sensorloc,name):
    name = os.path.join('simulationresults/optimalsensorplacement/', name)
    expsensorarray = np.reshape(sensorloc,(no_sensors,dim))
    np.save(name,expsensorarray)


def currscript_score(sensorloc): #To give a score function handle to the PSOSOLVER so that it can just ask for the value at a given SENSORLOC
    # and not bother about the parameters. It also helps calculate the expectation.
    i = 0
    print ('score called            ')
    score = 0
    gradient = 0
    for ele in avgspace:
        curr_obj = ops.optsenpmt(ele,no_sensors,dim,alpha,sigma)
        score+=curr_obj.score(sensorloc)
        # print(curr_obj.score(sensorloc));
        i+=1
    return 0 - score / i


def currscript_gradient(sensorloc): #Same as above,
    i = 0
    gradient = 0
    for ele in avgspace:
        curr_obj = ops.optsenpmt(ele,no_sensors,dim,alpha,sigma)
        gradient+=curr_obj.gradient_score(sensorloc)
        i += 1
    return 0-gradient/i


# def score_glb(sensorloc):
#     return currscript_score(sensorloc,avgspace,no_sensors,dim,alpha,sigma)
#
# def gradient_glb(sensorloc):
#     return
#
psol = pssol.PSO(currscript_score, 20, no_sensors*dim,.5,.5,1,-50+np.random.random(no_sensors*dim)*100,[-10*np.ones(no_sensors*dim),10*np.ones(no_sensors*dim)])

point = np.random.random(no_sensors*dim)
gdssol = gdslr.GDS(currscript_score,currscript_gradient,point)

rec_obj_pso = stlg.statelogger('optms_pso','optmspsolog',psol.curr_score,psol.globalmin,psol.current_pos)

rec_obj_gds = stlg.statelogger('optms_gds','optmsgdslog',gdssol.score_func(point))


for i in range(1):
    psol.update_pos()
    psol.update_currscores()
    psol.update_selfmin()
    psol.update_globalmin()
    psol.update_velocities()
    (centroid, spread) = psol.calc_swarm_props()

    print('\n\n\n')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  \n')
    print('PARTICLE SWARM OPTIMIZER REPORT')


    psol.report()

    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n')
    # print(psol.globalmin,psol.curr_score)
    rec_obj_pso.add_state(psol.curr_score,psol.globalmin,psol.current_pos)

    print('\n\n\n')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  \n')
   # print('GRADIENT DESCENT SOLVER REPORT')
   # curr_score = currscript_score(point)
   # point = gdssol.get_descented_point(point)
   # descented_score = currscript_score(point)
   # deltascore = descented_score - curr_score
   # curr_score = descented_score
   # gdssol.report()
   # rec_obj_gds.add_state(gdssol.score_func(point))




savesensorarray(psol.globalminlocation,'psosolsens.npy')
savesensorarray(point,'gdssolsens.npy')
"""
a = np.load('optms_pso.npy')
b = np.load('optms_gds.npy')
plt.plot(a[1])
plt.plot(b)
plt.show()	
"""
