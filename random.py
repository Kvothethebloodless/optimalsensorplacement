
from scipy.integrate import quad
from scipy.optimize import fsolve
from math import cos, sin, sqrt, pi

def circle_diff(t):
    dx = -sin(t)
    dy = cos(t)
    return sqrt(dx*dx+dy*dy)

def sin_diff(t):
	dx = 1
	dy = cos(t)
	return sqrt(dx*dx+dy*dy)

def curve_length(t0, S, length):
	return quad(S, 0, t0)[0] - length

def solve_t(curve_diff, length):
	return fsolve(curve_length, 0.0, (curve_diff, length))[0]

print solve_t(circle_diff, 2*pi)
print solve_t(sin_diff, 7.640395578)