import numpy as np
from scipy.misc import derivative


# import numpy.linalg as lin




class optsenpmt:
    def __init__(self, target_loc, no_sensors, dim, alpha, sigma):
        """

        :rtype: object
        """
        self.no_sensors = no_sensors
        self.target_loc = target_loc
        self.dim = dim
        self.alpha = alpha
        self.sigma = sigma
        

    def score(self, sensorlocarray):
        self.sensorlocarray = sensorlocarray
        self.allvectorcreate()
        return self.fim_det_alternate()

    def auxilary_score(self, *args):
        dummy_list = []
        for ele in args:
            dummy_list.append(ele)
        return self.score(np.array(dummy_list))

    def gradient_score(self, sensorlocarray):
        # self.sensorlocarray = sensorlocarray;
        h = 0.1e-5
        gradient = []
        gradient2 = []
        for a in range(self.dim * self.no_sensors):
            posdummy_sensorloc = np.copy(sensorlocarray)
            posdummy_sensorloc[a] += h
            # print(posdummy_sensorloc);
            negdummy_sensorloc = np.copy(sensorlocarray)
            negdummy_sensorloc[a] -= h
            # print(negdummy_sensorloc)
            partialgrad_a = (self.score(posdummy_sensorloc) - self.score(negdummy_sensorloc)) / (2 * h)
            partialgrad_a_1 = self.partial_derivative(self.auxilary_score, a, sensorlocarray)

            # print('This is the partial gradient')

            # print(self.score(posdummy_sensorloc)-self.score(negdummy_sensorloc))
            gradient.append(partialgrad_a)
            gradient2.append(partialgrad_a_1)
        return np.array(gradient2)

    @staticmethod
    def partial_derivative(func, var, point=[]):
        args = point[:]
 #TIME TAKING!!!
        def wraps(x):
            args[var] = x
            return func(*args)

        return derivative(wraps, point[var], dx=1e-6)
 #TIME TAKING!!!
    def summand(self, i, j):
        return .5 * (self.beta_2i[i ] * self.beta_2i[j ] * np.power(self.divv_dixdj(i , j ), 2))

    def ithsensor_loc(self, i):  # Expected sensorlocarray is 1byn [1,n] where n= ndim*no_sensors;
        ithsensor = self.sensorlocarray[(i - 1) * self.dim:(i * self.dim)]
        return ithsensor
    
    #TIME TAKING!!!
    def fim_det(self): #Taking a lot of time! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        fimdet = 0
        for i in range(self.no_sensors):
            for j in range(self.no_sensors):
                fimdet += self.summand(i, j)
                # print(self.summand(i,j))

        return fimdet
    def fim_det_alternate(self):
        self.beta_matrix = np.outer(self.beta_2i, np.transpose(self.beta_2i))
        divij = [ [ self.divv_dixdj(i,j) for j in range(self.no_sensors) ] for i in range(self.no_sensors)]
        divij = np.array(divij)
        self.divv_dixdj_matrix = divij**2 
        summand_matrix = np.multiply(self.beta_matrix,self.divv_dixdj_matrix)
        return .5* np.sum(summand_matrix)
        
 #TIME TAKING!!!
    def beta_2i_func(self, i):

        p1 = (np.power(self.alpha, 2)) / (2 * self.di_2[i])
        p2 = 1 / (np.power(self.sigma, 2) * self.di_alpha[i])

        return p1 + p2
 #TIME TAKING!!!
    def di_a_func(self, i, a):
        ith_sensor = self.ithsensor_loc(i)
        return np.power(np.linalg.norm(self.target_loc - ith_sensor), a)
 #TIME TAKING!!!
    def divv_di_func(self, i):  # divv_di is in the shape of 1by2 [1,n_dim]
        div_di = (self.target_loc - self.ithsensor_loc(i)) / (self.di_a(i, 1))
        return div_di
 #TIME TAKING!!!
    def allvectorcreate(self):
        """Creates vectors of all quantities dependent on i instead of calculating every time.
           Quantities of interest: 
               beta_2i
               divv_di
               di_a for a = 1
               di_a for a = 2
               di_a for a = alpha
        """
        self.di = np.empty(self.no_sensors)
        self.di_2 = np.empty_like(self.di)
        self.di_alpha = np.empty_like(self.di)
        self.divv_di = np.empty((self.no_sensors, self.dim)) #divv_di is of the same shape as s - 1 by ndim
        self.beta_2i = np.empty_like(self.di)


        for i in range(self.no_sensors):
            #Relative position of the target with respect to the ith sensor.
            self.targetrelpos = self.target_loc - self.ithsensor_loc(i+1) #i+1 because ithsensor_loc
            #works only from sensor 1. Sensor zero is meaningless in this context.
            self.di[i] =  np.linalg.norm(self.targetrelpos)
            self.di_2[i] = np.power(self.di[i], 2)
            self.di_alpha[i] = np.power(self.di[i], self.alpha)
            self.divv_di[i] = self.targetrelpos/self.di[i]
            self.beta_2i[i] = self.beta_2i_func(i)
        

        
    def divv_dixdj(self, i, j):
        return np.linalg.norm(np.cross(self.divv_di[i], self.divv_di[j]))












