import numpy as np
from copy import deepcopy

class FitnessFunction:

	def __init__( self, X_train, y_train, use_linear_scaling=True, error_metric='mse' ):
		self.X_train = X_train
		self.y_train = y_train
		self.use_linear_scaling = use_linear_scaling
		self.evaluations = 0
		self.error_metric = error_metric
		if error_metric == 'binary_acc':
			self.classes = np.unique(y_train)
			if len(self.classes) != 2:
				raise ValueError('error_metric set to binary accuracy but num classes is', len(self.classes))
		self.SampleTrainingSet()

	def SampleTrainingSet(self, revert_to_normal_training_set=False):
		if revert_to_normal_training_set:
			self.bootstrap_indices = np.array(range(len(self.y_train)))
		else:
			self.bootstrap_indices = np.array([ j for j in np.random.randint(len(self.y_train), size=len(self.y_train)) ])

		self.bootstrap_X = self.X_train[self.bootstrap_indices]
		self.bootstrap_y = self.y_train[self.bootstrap_indices]

		if self.use_linear_scaling:
			self.bootstrap_y_mean = np.mean(self.bootstrap_y)
			self.bootstrap_y_deviation = self.bootstrap_y - self.bootstrap_y_mean


	def ComputeError( self, y, p ):
		error = np.inf
		if self.error_metric == 'mse':
			error = np.mean(np.square(y - p))
		elif self.error_metric == 'binary_acc':
			error = np.mean(y != p)
		else:
			raise ValueError("error_metric not recognized:", self.error_metric)
		if np.isnan(error):
			error = np.inf
		return error


	def ComputeOutput(self, individual, X, adjust_linear_scaling=False, cache_output=True):
		output = individual.GetOutput( X )

		if self.use_linear_scaling:
			if adjust_linear_scaling:
				output_mean = np.mean(output)
				output_deviation = output - output_mean
				b = np.sum(np.multiply(self.bootstrap_y_deviation, output_deviation))/np.sum(np.square(output_deviation))
				if np.isnan(b):
					b = 0.0
				a = self.bootstrap_y_mean - b * output_mean
				individual.a = a
				individual.b = b
			output = individual.a + individual.b*output

		if self.error_metric == 'binary_acc':
			output[ output > .5 ] = 1.0
			output[ output <= .5 ] = 0.0

		if cache_output:
			individual.cached_output = output

		return output

	def Evaluate( self, individual ):

		self.evaluations = self.evaluations + 1
		individual.fitness = np.inf

		output = self.ComputeOutput(individual, self.bootstrap_X, 
			adjust_linear_scaling=self.use_linear_scaling, cache_output=True)

		individual.fitness = self.ComputeError( self.bootstrap_y, output )