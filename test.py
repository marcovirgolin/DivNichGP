# Libraries
import numpy as np 
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from copy import deepcopy

# Internal imports
from simplegp.Nodes.BaseNode import Node
from simplegp.Nodes.SymbolicRegressionNodes import *
from divnichgp.Evolution.Evolution import DivNichGP
from divnichgp.SKLearnInterface import DivNichGPRegressionEstimator as DNGP

# Set random seed
np.random.seed(42)

# Load regression dataset 
X, y = sklearn.datasets.load_boston( return_X_y=True ) #sklearn.datasets.load_diabetes(return_X_y=True) #
y_std = np.std(y)
X = scale(X)
y = scale(y)

# Take a dataset split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=42 )

dngp = DNGP(pop_size=100, radius=0, niche_size=1, max_generations=100, 
	verbose=True, max_tree_size=100,
	crossover_rate=0.475, mutation_rate=0.475, 
	initialization_max_tree_height=6, use_erc=True,
	tournament_size=4, max_features=-1, use_linear_scaling=True, 
	error_metric='mse',
	functions = [ AddNode(), SubNode(), MulNode(), DivNode() ])

dngp.fit(X_train,y_train)

print('Train RMSE:',  y_std * np.sqrt( np.mean(np.square(y_train - dngp.predict(X_train)))) )
print('Test RMSE:', y_std * np.sqrt( np.mean(np.square(y_test - dngp.predict(X_test)))) )