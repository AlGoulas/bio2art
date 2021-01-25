#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

import numpy as np

from bio2art import importnet
from echoes.tasks import MemoryCapacity
from echoes.plotting import plot_forgetting_curve, set_mystyle

# Compare memory performance in a lagged sequence memory task of a reservoir 
# with random topology and a reservoir with topology based on a biological 
# neuronal network

# IMPORTANT: ESN are depending on many parameters that can have a tremendous
# impact on the performance (e.g., activation function, weigh values of 
# Win and Wrec matrices, etc). So meaningful comparisons requeire a search
# over parameters. The example here is just to desmontrate use
# of the bio2art and esn, NOT for drawing final conclusions.

# Activation functions - can be used beyond the default tanh for the echo 
# state network 
def sigmoid(x):
   x = 1/(1+np.exp(-x)) 
   return x 

def relu(x):
    x = np.maximum(x, 0.)
    return x

# Return the density of the matrix X (density is a percentage of exisiting
# non-zero entries over all possible entries given such amtrix X) 
def density_matrix(X):    
    # Calculate the current density of the matrix
    # It includes the diagonal!
    X_size = X.shape
    non_zeros = np.where(X != 0)
    density = len(non_zeros[0]) / (X_size[0] * X_size[1])
    return density

# It threshold the conenctivity matrix X to satisfy the desired_density.
def threshold_matrix(X, desired_density):
    #Calculate the current density of the matrix
    #It includes the diagonal! 
    X_size = X.shape
    current_non_zeros = np.where(X != 0)
    
    current_density = len(current_non_zeros[0]) / (X_size[0] * X_size[1])
    
    #Clearly the operation makes sense 
    if(current_density <= desired_density):
        print("Current density smaller or equal than the desired one...")
    else: 
        desired_non_zeros = desired_density * (X_size[0] * X_size[1])
        
        nr_entries_to_set_to_zero = int(np.round(len(current_non_zeros[0]) - desired_non_zeros)) 
    
        current_non_zeros_rand_index = np.random.permutation(len(current_non_zeros[0]))
        
        x = current_non_zeros[0]
        y = current_non_zeros[1]
        
        x = x[current_non_zeros_rand_index[0:nr_entries_to_set_to_zero]]
        y = y[current_non_zeros_rand_index[0:nr_entries_to_set_to_zero]]
        
        X[(x,y)] = 0
        
    return X
    
# Convert connectome to matrix     

# Optional specification of inhomogenoous number of neurons. Comment out and 
# tailor accordingly. Otherwise ND=None
# ND_areas = np.random.choice([10, 8, 1], p=[.1, .1, .8], size=(57,))

#Specify here the folder where your connectomes are contained 
path_to_connectome_folder = Path('/Users/alexandrosgoulas/Data/work-stuff/python-code/packages/bio2art/connectomes/')#change to the folder where the Bio2Art was installed

# The neural network that we would like to use
data_name = 'Marmoset_Normalized'

neuron_density = np.zeros((55,), dtype=int)
neuron_density[:] = 2 

net_orig, net_scaled, region_neuron_ids = importnet.from_conn_mat(
    data_name = data_name, 
    path_to_connectome_folder = path_to_connectome_folder, 
    neuron_density = neuron_density, 
    target_sparsity = .1,
    intrinsic_conn = True, 
    target_sparsity_intrinsic = .5,
    rand_partition = True,
    keep_diag = True
    )

# Keep in this variable the size of the C_Neurons to intiialize the reservoir
size_of_matrix = net_scaled.shape[0]

# Echo state network memory
set_mystyle() # make nicer plots, can be removed

# Echo state network parameters (after Jaeger)
n_reservoir = size_of_matrix
W = np.random.choice([0, .47, -.47], p=[.5, .25, .25], size=(size_of_matrix, size_of_matrix))
W_in = np.random.choice([.1, -.1], p=[.5, .5], size=(n_reservoir, 2))

spectral_radius = .9

density_for_reservoir = density_matrix(net_scaled)
W = threshold_matrix(W, density_for_reservoir)

# Task parameters (after Jaeger)
inputs_func=np.random.uniform
inputs_params={'low':-.1, 'high':.1, 'size':300}
lags = [1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]

# Random reservoir
# Initialize the reservoir object
esn_params = dict(
    n_inputs=1,
    n_outputs=len(lags),  # automatically decided based on lags
    n_reservoir=size_of_matrix,
    W=W,
    W_in=W_in,
    spectral_radius=spectral_radius,
    bias=0, 
    n_transient=100,
    regression_method="pinv"
)

# Initialize the task object
mc = MemoryCapacity(
    inputs_func=inputs_func,
    inputs_params=inputs_params,
    esn_params=esn_params,
    lags=lags
).fit_predict()  # Run the task

#Plot the memory curve for the reservoir with random topology
plot_forgetting_curve(mc.lags, mc.forgetting_curve_)

# What we retouch for the bio2art network is W. To this end, get the unique 
# pair of values that the random reservoir was initialized with and replace
# the actual weights of the C_Neurons

# Get the indexes for the non zero elements of C_Neurons
non_zero_net_scaled = np.where(net_scaled != 0)

x_non_zero_C_Neurons = non_zero_net_scaled[0]
y_non_zero_C_Neurons = non_zero_net_scaled[1]

rand_indexes_of_non_zeros = np.random.permutation(len(x_non_zero_C_Neurons))

indexes_for_unique1 = int(np.floor(len(rand_indexes_of_non_zeros)/2))

# Assign the same weight as for the random reervoir for a comparison
net_scaled[(x_non_zero_C_Neurons[rand_indexes_of_non_zeros[:indexes_for_unique1]], 
            y_non_zero_C_Neurons[rand_indexes_of_non_zeros[:indexes_for_unique1]])] = .47

net_scaled[(x_non_zero_C_Neurons[rand_indexes_of_non_zeros[indexes_for_unique1:]], 
            y_non_zero_C_Neurons[rand_indexes_of_non_zeros[indexes_for_unique1:]])] = -.47


# Bio connectome reservoire 
# Initialize the reservoir
esn_params = dict(
    n_inputs=1,
    n_outputs=len(lags),  # automatically decided based on lags
    n_reservoir=size_of_matrix,
    W=net_scaled,
    W_in=W_in,
    spectral_radius=spectral_radius,
    bias=0,
    n_transient=100,
    regression_method="pinv"
)

# Initialize the task object 
mc_bio = MemoryCapacity(
    inputs_func=inputs_func,
    inputs_params=inputs_params,
    esn_params=esn_params,
    lags=lags
).fit_predict()  # Run the task

# Plot the memory curve for the reservoir with the biological connectome 
# topology
plot_forgetting_curve(mc_bio.lags, mc_bio.forgetting_curve_)
      
