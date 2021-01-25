#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error

from echoes import ESNPredictive
from bio2art import importnet

sns.set(context="notebook", style="whitegrid", font_scale=1.4, 
        rc={'grid.linestyle': '--', 
            'grid.linewidth': 0.8,})

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
    # It included the diagonal!
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

# Generate pattern to memorize with length N and from a uniform distribution
# between low and high values.   
# Trials have a memorize period (the generated numbers=pattern_length) 
# and a recall period, that is, 0s=pattern_length. The trials are padded with
# zeros and ones with 1 denoting "recall cue". Thus, trials are 2D arrays.    
def generate_input_output_patterns(pattern_length=3, low=0., high=1., nr_of_trials=100):
    all_input_trials = None
    all_output_trials = None
    
    for tr in range(nr_of_trials):
        # Create here standard blocks of the trials, namely the cue and "null input"
        # The cue is a 1 on a channel that is not used for the patterns,
        # so concatanate a vector with 0s when we have a trial with input 
        # (the patterns to be memomorized) and a 1 to denote the recall cue
        # when the reservoir has to replay the patterns. 
        
        # 1 is presented only once, with zeros following it for the "null input" 
        
        null_input = np.zeros((2, pattern_length+1))
        
        # Assign the cue at the upper left corner so that the first column of the 
        # null input is actually the recall cue.
        null_input[0,0] = 1
        
        padding_for_trial = np.zeros((pattern_length,))
        
        #Generate one trial based on the specifications
        trial = np.random.uniform(low, high, pattern_length)
    
        # Add the padding that corresponds to a cue=0 (that means no replaying yet,
        # but leanrning the input patterns)
        trial = np.vstack((padding_for_trial, trial))
        
        input_trial = np.hstack((trial, null_input))
        
        # Now we can construct the desired ouput. This is basically a "mirrored"
        # version of the input, so construct accordingly: where null_input put
        # the current trial and vice versa. 
        
        # We need no padding for the output (no "cue needed"). Just require 0s
        # when the pattern is being learned.
        null_output = np.zeros((1, pattern_length+1))#Add 1 column to have the same length with input
        
        trial = trial[1:,:]
        
        output_trial = np.hstack((null_output, trial))
        
        # Concatanate the generated input/output trials to the the overall 
        # trials array 
        if all_input_trials is None:
            all_input_trials = input_trial
            all_output_trials = output_trial
        else:
            all_input_trials = np.hstack((all_input_trials, input_trial))
            all_output_trials = np.hstack((all_output_trials, output_trial))
        
    all_input_trials = all_input_trials.T
    all_output_trials = all_output_trials.T     
    
    return all_input_trials, all_output_trials 


def evaluate_performance(predicted, actual, discard=0, low=0., high=1.):
    if discard > 0:
        actual = actual[discard:]
        predicted = predicted[discard:]
    
    # Use the fact that the positions of interest are the non 0s in
    # the actual array. This is task specific so if the the format
    # of the trials changes then the way to fetch elements also must change!!
    # import numpy as np 
    
    indexes_not_zeros = np.where(actual != 0)[0]  
    
    predicted = predicted[indexes_not_zeros]
    actual = actual[indexes_not_zeros]
    
    err = mean_squared_error(actual, predicted)
    
    # Generate a sequence from the same distribution used for the trials
    # This will function as a "null" baseline
    predicted_rand = np.random.uniform(low, high, len(predicted))
    err_null_random = mean_squared_error(actual, predicted_rand)
    
    return err, err_null_random, actual, predicted, predicted_rand

# Plot trials for visual inspection of the performance
def plot_trials(trial_nrs, trial_length, actual, predicted, predicted_rand, mean_trial_predictor):
    
    for i,tr in enumerate(trial_nrs):
        plt.figure(figsize=(15, 4))
        
        start = (tr-1) * trial_length
        stop = start + trial_length
        
        plt.plot(actual[start:stop], 
                  label='actual_trials',
                  color="steelblue", 
                  linewidth=5.5)
        
        plt.plot(predicted[start:stop],
                  label='predicted signal',
                  linestyle='--',
                  color="orange", 
                  linewidth=2,)
        
        plt.plot(predicted_rand[start:stop],
                  label='predicted randomly',
                  linestyle='--',
                  color="green", 
                  linewidth=2,)
        
        plt.plot(mean_trial_predictor[start:stop],
                  label='predicted from mean',
                  linestyle='--',
                  color="magenta", 
                  linewidth=2,)
        
        plt.ylabel("values")
        plt.xlabel('trials across time')
        plt.legend(fontsize=("small"), loc=2)
        
        
#Get the mean of the input trials. This will return a null baseline as it
#is usually the case, that is, "predictions" are just the mean of the 
#observations.
def get_mean_of_trials(input_trials_train, pattern_length, discard=0):  
    if discard > 0:
        input_trials_train = input_trials_train[discard:, :]
    
    indexes_trials = np.where(input_trials_train[:,1] !=0 )[0]
    
    trials = input_trials_train[indexes_trials, 1]
    
    nr_trials_to_consider = len(trials) / pattern_length
    
    #Initialize array of trial means
    mean_trial_predictor = None
    
    #Note that we have already selected only the trials and not the 0s and cue
    #signal. Thus, trial length = pattern_length
    
    start_index = 0
    stop_index = pattern_length
    
    for tr in range(int(nr_trials_to_consider)):
        current_trial = trials[start_index:stop_index] 
        
        mean_current_trial = np.mean(current_trial)
        
        current_mean_predictor = np.full((pattern_length, ), mean_current_trial)
        
        if mean_trial_predictor is None:
            mean_trial_predictor = current_mean_predictor
        else:
            mean_trial_predictor = np.hstack((mean_trial_predictor, current_mean_predictor))
        
        #Update indexes to grab the next trial
        start_index = start_index + pattern_length
        stop_index = stop_index + pattern_length
    
    return mean_trial_predictor        

#Run the memory tests with a random and a bio instantiation ESN

# Amount of numbers to be memorized in each trial
pattern_length=5

input_trials_train, output_trials_train = generate_input_output_patterns(
    pattern_length=pattern_length, 
    low=0., 
    high=1., 
    nr_of_trials=1000
    )

input_trials_test, output_trials_test = generate_input_output_patterns(
    pattern_length=pattern_length, 
    low=0., 
    high=1., 
    nr_of_trials=1000
    )

# Determine the low and high limits of a uniform distribution from which the
# weights for the reservoire willl be generated
weight_reservoire_high = 1.
weight_reservoire_low = -1.

# Biological topology reservoire
# Specify here the folder where your connectomes are contained 
path_to_connectome_folder = Path('/Users/.../Bio2Art/connectomes/')#change to the folder where the Bio2Art was installed

# Specify here the connectome that we will use. In this example we use the
# macaque monkey connectome. 
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

# What we retouch for the bio2art network is W. To this end, get the unique 
# pair of values that the random reservoir was initialized with and replace
# the actual weights of the C_Neurons

# Get the indexes for the non zero elements of C_Neurons
non_zero_net_rescaled = np.where(net_scaled != 0)

x_non_zero_net_rescaled = non_zero_net_rescaled[0]
y_non_zero_net_rescaled = non_zero_net_rescaled[1]

rand_indexes_of_non_zeros = np.random.permutation(len(x_non_zero_net_rescaled))

indexes_for_unique1 = int(np.floor(len(rand_indexes_of_non_zeros)/2))

# Random reservoire
# Parameters and initialization of the random reservoire
trials_out = 20
n_transient = ((pattern_length*2)+1)*trials_out

n_reservoir = net_scaled.shape[0]

W = np.random.uniform(weight_reservoire_low, 
                      weight_reservoire_high, 
                      [n_reservoir, n_reservoir])

# Equate density of random and bio topology reservoirs
density_for_reservoir = density_matrix(net_scaled)
W = threshold_matrix(W, density_for_reservoir)

# Fill in the weights of the bio reservoire with the exact values used for 
# the random
weights_random_reservoire = W[np.where(W != 0)]

# Assign them to the topology of the bio resevoire
bio_weights_index = np.where(net_scaled != 0)
net_scaled[bio_weights_index] = weights_random_reservoire

# Train and test on the random reservoire
esn = ESNPredictive(
    n_inputs=2,
    n_outputs=1,
    n_reservoir=n_reservoir,
    W=W,
    spectral_radius=1.,
    leak_rate=.6,
    n_transient=n_transient,
    teacher_forcing=False,
    activation=relu,
    regression_method="pinv"
)

esn.fit(input_trials_train, output_trials_train)

prediction_test = esn.predict(input_trials_test)

# No need to generate a seperate err_null_random output. Thus,
# use the same output argument for the random and bio reservoir
# that is: err_null_random, predicted_rand
(err_r, err_null_random, 
 actual, predicted, predicted_rand) = evaluate_performance(prediction_test, 
                                                           output_trials_test, 
                                                           low=0.,
                                                           high=1.,
                                                           discard=n_transient
                                                           )

# Compute the error with the null mean trial predictor as well.

# Construct a null mean trial predictor
mean_trial_predictor = get_mean_of_trials(input_trials_train, 
                                           pattern_length, 
                                           discard=n_transient)
    
err_null_mean = mean_squared_error(actual, mean_trial_predictor)

#Plot some trials
plot_trials([1,10,100], 
            pattern_length, 
            actual, 
            predicted, 
            predicted_rand,
            mean_trial_predictor)

#Train and test on the bio topology reservoire
esn = ESNPredictive( 
    n_inputs=2,
    n_outputs=1,
    n_reservoir=n_reservoir,
    W=net_scaled,
    spectral_radius=1.,
    leak_rate=.6,
    n_transient=n_transient,
    teacher_forcing=False,
    activation=relu,
    regression_method="pinv"
)

esn.fit(input_trials_train, output_trials_train)

prediction_test = esn.predict(input_trials_test)

(err_bio, err_null_random, 
 actual, predicted, predicted_rand) = evaluate_performance(prediction_test, 
                                                           output_trials_test, 
                                                           low=0.,
                                                           high=1.,
                                                           discard=n_transient)

#Plot some trials
plot_trials([1,10,100], 
            pattern_length, 
            actual,
            predicted,
            predicted_rand,
            mean_trial_predictor)


