import numpy as np
from scipy.misc import derivative

import matplotlib.pyplot as plt
#import numpy.linalg as lin




class optsenpmt():
    def __init__(self,target_loc,no_sensors,dim,alpha,sigma):
        """

        :rtype: object
        """
        self.no_sensors = no_sensors;
        self.target_loc = target_loc;
        self.dim = dim;
        self.alpha = alpha;
        self.sigma = sigma;


    def score(self,sensorlocarray):
        self.sensorlocarray = sensorlocarray;
        return self.fim_det();

    def auxilary_score(self,*args):
        dummy_list = []
        for ele in args:
            dummy_list.append(ele)
        return self.score(np.array(dummy_list))

    def gradient_score(self,sensorlocarray):
        # self.sensorlocarray = sensorlocarray;
        h = 0.1e-5;
        gradient = [];
        gradient2 = [];
        for a in range(self.dim*self.no_sensors):
            posdummy_sensorloc = np.copy(sensorlocarray);
            posdummy_sensorloc[a] += h;
            # print(posdummy_sensorloc);
            negdummy_sensorloc = np.copy(sensorlocarray);
            negdummy_sensorloc[a] -= h;
            # print(negdummy_sensorloc)
            partialgrad_a = (self.score(posdummy_sensorloc)-self.score(negdummy_sensorloc))/(2*h);
            partialgrad_a_1 = self.partial_derivative(self.auxilary_score,a,sensorlocarray)

            # print('This is the partial gradient')

            # print(self.score(posdummy_sensorloc)-self.score(negdummy_sensorloc))
            gradient.append(partialgrad_a)
            gradient2.append(partialgrad_a_1)
        return np.array(gradient2)

    def partial_derivative(self,func, var, point=[]):
        args = point[:]
        def wraps(x):
            args[var] = x
            return func(*args)
        return derivative(wraps, point[var], dx = 1e-6)


    def summand(self,i,j):
        return .5*(self.beta_2i(i+1)*self.beta_2i(j+1)*np.power(self.divv_dixdj(i+1,j+1),2));



    def ithsensor_loc(self,i): #Expected sensorlocarray is 1byn [1,n] where n= ndim*no_sensors;
        ithsensor = self.sensorlocarray[(i-1)*self.dim:(i*(self.dim))];
        return ithsensor


    def fim_det(self):
        fimdet = 0;
        for i in range(self.no_sensors):
            for j in range(self.no_sensors):
                fimdet += self.summand(i,j);
                # print(self.summand(i,j))

        return fimdet;


    def beta_2i(self,i):

        p1 = (np.power(self.alpha,2))/(2*self.di_a(i,2));
        p2 = 1/(np.power(self.sigma,2)*self.di_a(i,self.alpha))

        
        return p1+p2;

    def di_a(self,i,a):
        ith_sensor = self.ithsensor_loc(i)
        return np.power(np.linalg.norm(self.target_loc-ith_sensor),a);


    def divv_di(self,i): #divv_di is in the shape of 1by2 [1,n_dim]
        div_di = (self.target_loc - self.ithsensor_loc(i))/(self.di_a(i,1));
        return div_di;

    def divv_dixdj(self,i,j):
        return np.linalg.norm(np.cross(self.divv_di(i),self.divv_di(j)));



