#import matplotlib.pyplot as plt

class GDS:
	
	def __init__(self,score_func,gradient_score_func,in_point):
		self.score_func = score_func
		self.gradient_func = gradient_score_func
		self.accel_coeff = .5
		self.new_point = in_point

	# self.sensor_arena = sensor_arena

	def compute_gradient(self,point):
		self.grad = self.gradient_func(point)
	def descent(self,point):
		self.new_point = point - (self.grad*self.accel_coeff)
	def get_descented_point(self,point):
		self.point =point
		self.compute_gradient(point)
		self.descent(point)
		return  self.new_point
	def report(self):
		print("The solver is currently at "+ str(self.point) +"\n")
		# print(self.sensor_arena.get_score(self.point))
		print(self.score_func(self.point))
		print("The current score is " + str(self.score_func(self.point))+ "\n")
		print("The computed gradient is " + str(self.grad) + "\n")
		print("The point moved to " +str(self.new_point)+"\n")
		print("The new score is" + str(self.score_func(self.new_point))+"\n")
		
		
		
	