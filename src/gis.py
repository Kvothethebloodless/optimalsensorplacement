import numpy as np
import pdb

import googlemaps
from geopy.distance import vincenty


class geography():
	def __init__(self):
		self.authkey = 'AIzaSyApbqhgLUF-g2bIlkzKTYg7Ou1cjEF5FQk'
		self.gmaps = googlemaps.Client(key=self.authkey)

	def rect_map(self, (latup, latdown, longleft, longright), no_points=1024):
		# Returns a rectangular grid of latitude and longitude points.
		self.no_points = no_points
		latvector = np.linspace(latup, latdown, no_points)
		longvector = np.linspace(longleft, longright, no_points)
		(self.longmatrix, self.latmatrix) = np.meshgrid(longvector, latvector)
		return (self.latmatrix, self.longmatrix)

	def converttomts(self):
		pdb.set_trace()
		origin = (self.latmatrix[0, 0], self.longmatrix[0, 0])
		self.x_mt = np.empty_like(self.latmatrix)
		self.y_mt = np.empty_like(self.x_mt)
		for i in range(self.no_points):
			for j in range(self.no_points):
				point = (self.latmatrix[i, j], self.longmatrix[i, j])
				self.x_mt[i, j] = vincenty((self.latmatrix[i, 0], self.longmatrix[i, 0]), point).meters
				self.y_mt[i, j] = vincenty((self.latmatrix[0, j], self.longmatrix[0, j]), point).meters
		return (self.x_mt, self.y_mt)

	def snaptoroad_latlong(self):
		fp = open('simulationdata/GISdata/road_latlong.txt', 'r')
		data = fp.readlines()
		data2 = [ele.split(', ') for ele in data]
		cords_list = [[float(ele[0]), float(ele[1])] for ele in data2]

		return
