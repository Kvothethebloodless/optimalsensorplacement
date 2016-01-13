import logging
import numpy as    np


class statelogger():
	def __init__(self,datafilename,logfilename,*args):
		#datafilename = filename to log the state data.
		#logfilename = filename to log statements.
		self.no_variables = len(args);
		self.filename = datafilename+str('.npy');
		self.statenumber = 1;

		self.all_states = []
		self.logfile = logfilename
		a = open(self.logfile,'w+')
		a.close()

		self.logger = logging.getLogger("statelogger")

		self.logger.setLevel(logging.DEBUG) #Set to Debug only to print every function call

		self.fh = logging.FileHandler(self.logfile)
		formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
		self.fh.setFormatter(formatter)
		self.logger.addHandler(self.fh)

		self.logger.info('**********STARTING***********')
		self.logger.info('Logging to - *' + str(self.logfile) + '*')
		self.logger.info('Recieved ' + str(self.no_variables) + ' variables')
		self.logger.info('Starting Recording')

		self.current_state = []

		for i in range(len(args)):
			self.current_state.append(np.array(args[i]));
			self.all_states.append([]) #Created a list of lists.
			self.all_states[i].append(np.array(args[i])) #We are making a list of lists which the all_states is going to be in this script. While being written to file it is cast into
			#an numpy array. Until then each list is tracking one variable, appending the new value each time. This helps to not lose matrix inputs which loose their granularity when
			#cast into numpy array by vstacking them with the previous entries.
			#Example: Vstacking [[1,2],[3,4]] and itself will not result in [[[1,2],[3,4]],[[1,2],[3,4]]] but [[1,2],[3,4],[1,2],[3,4]] which is unintended.
		#self.all_states = self.current_state

		self.logger.info('Initialization Finished')

	def update_data(self):
		logger_update_data = self.logger.getChild('update_data')

		for i in range(self.no_variables):
			self.all_states[i].append(self.current_state[i]) #Appending the new value to its list value.
		logger_update_data.debug('Data Update successful')

	def write_to_file(self):
		logger_write_to_file  = self.logger.getChild('write_to_file')
		np.save(self.filename,np.array(self.all_states))#Overwriting the existing file with new data series including the latest appended data point.
		logger_write_to_file.debug('Written to file succesfully')

	def add_state(self,*args):
		"""

		:rtype: Does not return anything. Adds State
		"""
		logger_new_state = self.logger.getChild('add_state')

		logger_new_state.info('Recieved %s variables.' %(len(args)))
		self.statenumber = self.statenumber+1
		self.current_state = [];

		try:
			for i in range(len(args)):
				self.current_state.append(np.array(args[i]));
			self.update_data()
		except(ValueError,IndexError):
			print('Inputted number of variables does not meet exisiting bank requirement');
			logger_new_state.error('Inputted number of variables does not meet exisiting bank requirement of %s' %(self.no_variables))
		else:
			self.write_to_file()
			logger_new_state.info('Added %s variables to bank' %(len(args)))
	def close_logger(self):
		self.logger.info('**********FINISEHD***********')

		self.logger.removeHandler(self.fh)
	#def report_log_io():
