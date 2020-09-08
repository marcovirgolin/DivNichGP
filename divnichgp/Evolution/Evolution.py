import numpy as np
from numpy.random import random, randint, shuffle
import time
from copy import deepcopy
from scipy.stats import mode


from simplegp.Variation import Variation
from simplegp.Selection import Selection


class DivNichGP:

	def __init__(
		self,
		fitness_function,
		functions,
		terminals,
		pop_size=500,
		crossover_rate=0.5,
		mutation_rate=0.5,
		max_evaluations=-1,
		max_generations=-1,
		max_time=-1,
		initialization_max_tree_height=4,
		max_tree_size=100,
		max_features=-1,
		tournament_size=4,
		radius=0,
		niche_size=1,
		verbose=False
		):

		self.pop_size = pop_size
		self.fitness_function = fitness_function
		self.functions = functions
		self.terminals = terminals
		self.crossover_rate = crossover_rate
		self.mutation_rate = mutation_rate

		self.max_evaluations = max_evaluations
		self.max_generations = max_generations
		self.max_time = max_time

		self.initialization_max_tree_height = initialization_max_tree_height
		self.max_tree_size = max_tree_size
		self.max_features = max_features
		self.tournament_size = tournament_size

		self.radius = radius
		self.niche_size = niche_size

		self.generations = 0
		self.verbose = verbose
		self.population = []
		self.ensemble = []


	def __ShouldTerminate(self):
		must_terminate = False
		elapsed_time = time.time() - self.start_time
		if self.max_evaluations > 0 and self.fitness_function.evaluations >= self.max_evaluations:
			must_terminate = True
		elif self.max_generations > 0 and self.generations >= self.max_generations:
			must_terminate = True
		elif self.max_time > 0 and elapsed_time >= self.max_time:
			must_terminate = True

		if must_terminate and self.verbose:
			print('Terminating at\n\t', 
				self.generations, 'generations\n\t', self.fitness_function.evaluations, 'evaluations\n\t', np.round(elapsed_time,2), 'seconds')

		return must_terminate


	def ComputeDistance(self, individual1, individual2):
		dist = np.sqrt(np.sum(np.square(individual1.cached_output - individual2.cached_output)))
		return dist

	def Clearing(self):
		self.population = sorted(self.population, key=lambda x : x.fitness)
		for i in range(len(self.population)):
			if self.population[i].fitness < np.inf:
				n = 1
				for j in range(i+1, len(self.population)):
					if self.population[j].fitness < np.inf and self.ComputeDistance(self.population[i], self.population[j]) < max(self.radius, 1e-10):
						if n < self.niche_size:
							n += 1
						else:
							self.population[j].fitness = np.inf

	def GetEnsemblePredictions(self, ensemble, X):
		member_predictions = []
		for j in range(len(ensemble)):
			prediction = self.fitness_function.ComputeOutput(ensemble[j], X, 
				adjust_linear_scaling=False, cache_output=False)
			member_predictions.append(prediction)
		member_predictions = np.array(member_predictions)

		if self.fitness_function.error_metric == "binary_acc":
			ep = mode(member_predictions)[0][0]
		else:
			ep = np.mean(member_predictions, axis=0)
		return ep


	def GreedyEnsemble(self):

		best_ensemble_error = np.inf
		best_ensemble = []

		# the paper generates a validation set, we use the latest out-of-bag samples
		indices_used_in_last_training_set = np.unique(self.fitness_function.bootstrap_indices)
		all_indices = np.arange(len(self.fitness_function.y_train))
		out_of_bag_indices = np.setdiff1d(all_indices, indices_used_in_last_training_set)
		X_validation = self.fitness_function.X_train[out_of_bag_indices]
		y_validation = self.fitness_function.y_train[out_of_bag_indices]

		for i in range(len(self.population)):
			in_niche = False
			for j in range(i+1, len(self.population)):
				dist_ij = self.ComputeDistance(self.population[i], self.population[j])
				if dist_ij <= max(self.radius, 1e-10):
					in_niche = True
					break
			if in_niche == False:
				candidate_ensemble = best_ensemble + [self.population[i]]
				candidate_ensemble_prediction = self.GetEnsemblePredictions(candidate_ensemble, X_validation)
				candidate_ensemble_error = self.fitness_function.ComputeError(candidate_ensemble_prediction, y_validation)

				if candidate_ensemble_error < best_ensemble_error:
					best_ensemble = candidate_ensemble
					best_ensemble_error = candidate_ensemble_error

		return best_ensemble



	def Run(self):

		self.start_time = time.time()

		# ramped half-n-half initialization w/ rejection of duplicates
		self.population = []

		attempts_duplicate_rejection = 0
		max_attempts_duplicate_rejection = self.pop_size * 10
		already_generated_trees = set()

		half_pop_size = int(self.pop_size/2)
		for j in range(2):

			if j == 0:
				method = 'full'
			else:
				method = 'grow'

			curr_max_depth = 2
			init_depth_interval = self.pop_size / (self.initialization_max_tree_height - 1) / 2
			next_depth_interval = init_depth_interval

			i = 0
			while len(self.population) < (j+1)*half_pop_size:

				if i >= next_depth_interval:
					next_depth_interval += init_depth_interval
					curr_max_depth += 1

				t = Variation.GenerateRandomTree( self.functions, self.terminals, curr_max_depth, curr_height=0, method=method )
				t_as_str = str(t.GetSubtree())
				if t_as_str in already_generated_trees and attempts_duplicate_rejection < max_attempts_duplicate_rejection:
					del t
					attempts_duplicate_rejection += 1
					continue
				else:
					already_generated_trees.add(t_as_str)
					t.requires_reevaluation=True
					self.population.append( t )
					i += 1


		# Sample a training set
		self.fitness_function.SampleTrainingSet()

		# Evaluate
		for t in self.population:
			self.fitness_function.Evaluate(t)
				
		# Run generational loop	
		while not self.__ShouldTerminate():
			''' 
			It looks like the paper uses (mu,lambda)-evolution
			'''

			# Clearing method
			self.Clearing()

			# Evolve
			self.population = Selection.TournamentSelect( self.population, self.pop_size, tournament_size=self.tournament_size )
			O = []
			
			for i in range( self.pop_size ):

				o = deepcopy( self.population[i] )

				r = np.random.random()
				if (r < self.crossover_rate + self.mutation_rate):
					if (r < self.crossover_rate ):
						o = Variation.SubtreeCrossover( o, self.population[ randint( self.pop_size ) ] )
					else:
						o = Variation.SubtreeMutation( o, self.functions, self.terminals, max_height=self.initialization_max_tree_height )

					# check constraints
					if (self.max_tree_size > -1 and len(o.GetSubtree()) > self.max_tree_size):
						o = deepcopy(self.population[i])

				O.append(o)

			# The offspring population replaces the parent population
			self.population = O

			# Sample new training set
			self.fitness_function.SampleTrainingSet()

			# Evaluate
			for t in self.population:
				self.fitness_function.Evaluate(t)

			self.generations = self.generations + 1
			best_err = sorted(self.population, key=lambda x: x.fitness)[0].fitness

			if self.verbose:
				print ('g:', self.generations, 'current best error:', np.round(best_err,3))

		# Create final ensemble
		self.ensemble = self.GreedyEnsemble()