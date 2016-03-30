import numpy as np
import cv2 
import os 
import solvers.psosolver as pso
import utilities.statelogger as stl
import matplotlib.pyplot as plt

def gauss_kern(size, sizey=None):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = scipy.mgrid[-size:size+1, -sizey:sizey+1]
    g = scipy.exp(-(x**2/float(size)+y**2/float(sizey)))
    return g / g.max()


def score1(X):
    return 10*(X-[10,10])**2

def score2(X):
    return (X-[20,10])**2

def score3(X):
    return 20*(X-[10,20])**2

def score4(X):
    return 30*(X-[20,20])**2

def trutharray(x):
    ''' Returns a no_functions * no_eleinx matrix with truth value of each partition for applying it
    to a element'''
    tarray = [[],[],[],[]]
    for ele in x:
        if (ele[0]<20 & ele[1]<20):
            print('1')
            tarray[0].append(True)
            tarray[1].append(False)
            tarray[2].append(False)
            tarray[3].append(False)
        if ((ele[0]>=20&ele[0]<=40)&( ele[1]<20)):
            print('2')
            tarray[1].append(True)
            tarray[0].append(False)
            tarray[2].append(False)
            tarray[3].append(False)
        if ((ele[0]<20)&(ele[1]>=20&ele[1]<40)):
            print('3')
            tarray[2].append(True)
            tarray[1].append(False)
            tarray[0].append(False)
            tarray[3].append(False)  
        if ((ele[0]>=20&ele[0]<=40) & (ele[1]>=20&ele[1]<40)):
            print('4')
            tarray[3].append(True)
            tarray[1].append(False)
            tarray[2].append(False)
            tarray[0].append(False) 
    return tarray

def alt_trutharray(x):
    t1 = np.all(x<[20,20], axis = 1)
    t2 = np.all(x>=[20 ,0], axis = 1 ) & np.all(x<=[40,20], axis =1 )
    t3 = np.all(x>=[0,20], axis = 1) & np.all(x<=[20,40], axis = 1)
    t4 = np.all(x>=[20,20], axis = 1) & np.all(x<=[40,40], axis = 1)
    return [t1,t2,t3,t4]


def funcdef(x):
    return np.piecewise(x, alt_trutharray(x), [score1, score2, score3, score4])


