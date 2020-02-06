#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 14:34:29 2020

@author: alexandrosgoulas
"""

#from matplotlib import pyplot as plt
import numpy as np

import seaborn as sns
#from sklearn.metrics import mean_squared_error

from echoes import ESNPredictive

sns.set(context="notebook", style="whitegrid", font_scale=1.4, 
        rc={'grid.linestyle': '--', 
            'grid.linewidth': 0.8,})


def sigmoid(x):
    
   x = 1/(1+np.exp(-x))
     
   return x 

#Define some usuful functions for matrix density and a custom sigmoid
#to be optionally used as an activation function.

def density_matrix(X):
    
    import numpy as np
    
    #Calculate the current density of the matrix
    #It included the diagonal!
    X_size = X.shape
    non_zeros = np.where(X != 0)
    
    density = len(non_zeros[0]) / (X_size[0] * X_size[1])
    
    return density



def threshold_matrix(X, desired_density):
    
    import numpy as np
    
    #Calculate the current density of the matrix
    #It included the diagonal!
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


def GenerateInputOutputPatterns_Binary(pattern_length=3, nr_of_patterns=5, nr_of_trials=100):
    
    from random import randint
    import  numpy as np    
    
    #One way to go: intialize and index in the loop
    
    #Initialize the arrays to store all the trials
    #For the input: x = pattern_length+1
    #               y = ((nr_of_patterns*2)+1)*nr_of_trials
    
    #For the output:x = pattern_length
    #               y = ((nr_of_patterns*2)+1)*nr_of_trials
    
    #size_y = ((nr_of_patterns * 2) + 1) * nr_of_trials
    
    #all_input_trials = np.zeros((pattern_length+1, size_y))
    #all_output_trials = np.zeros((pattern_length, size_y))
    
    #Another way: initialize the arrays to keeep all input output with None
    #and concatanate the geenrate trials in the loop in each step.
        
    all_input_trials = None
    #all_ouput_trials = None
    
    for tr in range(0, nr_of_trials):

        #Create here standard blocks of the trials, namely the cue and "null input"
        #The cue is a 1 on a channel that is not used for the patterns,
        #so concatanate a vector with 0s when we have a trial with input 
        #(the patterns to be memomorized) and a 1 to denote the recall cue
        #when the reservoir has to replay the patterns. 
        
        #1 is presented only once, with zeros following it for the "null input" 
        
        null_input = np.zeros((pattern_length+1, nr_of_patterns+1))
        
        #Assign the cue at the upper left corner so that the first column of the 
        #null input is actually the recall cue.
        null_input[0,0] = 1
        
        padding_for_trial = np.zeros((nr_of_patterns,))
        
        #Generate one trial based on the specifications(pattern_length, nr_of_patterns)
        trial = np.zeros((pattern_length, nr_of_patterns))
        
        x = [randint(0, pattern_length-1) for r in range(0, nr_of_patterns)]
        y = [r for r in range(0, nr_of_patterns)]
        
        trial[(x,y)] = 1
    
        #Add the padding that corresponds to a cue=0 (that means no replaying yet,
        #but leanrning the input patterns)
        trial = np.vstack((padding_for_trial, trial))
        
        input_trial = np.hstack((trial, null_input))
        
        #Now we can construct the desired ouput. This is basically a "mirrored"
        #version of the input, so construct accordingly: where null_input put
        #the current trial and vice versa. 
        
        #We need no padding for the output (no "cue needed"). Just reqiuire 0s
        #when the pattern is being learned. 
        
        #TODO:The 0s as output in-between the cues have to be discardeed so that 
        #performance and weights is solely determined only by the memory performance
        #and not the ability to "fixate" (produce 0s) in the learning.
        
        null_output = np.zeros((pattern_length, nr_of_patterns+1))#Add 1 column to have the same length with input
        
        trial = trial[1:,:]
        
        output_trial = np.hstack((null_output, trial))
        
        #Concatanate the generate input/output trials to the the overall 
        #trials array 
        
        if all_input_trials is None:
            
            all_input_trials = input_trial
            all_output_trials = output_trial
            
        else:
            
            all_input_trials = np.hstack((all_input_trials, input_trial))
            all_output_trials = np.hstack((all_output_trials, output_trial))
        
      
    all_input_trials = all_input_trials.T
    all_output_trials = all_output_trials.T
        
    all_input_trials = all_input_trials.astype(int)
    all_output_trials = all_output_trials.astype(int)
    
    return all_input_trials, all_output_trials


#Generate pattern to memorize with length N and from distribution distr
    
def GenerateInputOutputPatterns_Cont(pattern_length=3, low=0., high=1., nr_of_trials=100):
    
    all_input_trials = None
    all_output_trials = None
    
    import numpy as np
    
    for tr in range(0, nr_of_trials):

        #Create here standard blocks of the trials, namely the cue and "null input"
        #The cue is a 1 on a channel that is not used for the patterns,
        #so concatanate a vector with 0s when we have a trial with input 
        #(the patterns to be memomorized) and a 1 to denote the recall cue
        #when the reservoir has to replay the patterns. 
        
        #1 is presented only once, with zeros following it for the "null input" 
        
        null_input = np.zeros((2, pattern_length+1))
        
        #Assign the cue at the upper left corner so that the first column of the 
        #null input is actually the recall cue.
        null_input[0,0] = 1
        
        padding_for_trial = np.zeros((pattern_length,))
        
        #Generate one trial based on the specifications(pattern_length, nr_of_patterns)
        #trial = np.zeros((pattern_length, nr_of_patterns))
        
        #x = [randint(0, pattern_length-1) for r in range(0, nr_of_patterns)]
        #y = [r for r in range(0, nr_of_patterns)]
        
        #trial[(x,y)] = 1
    
        trial = np.random.uniform(low, high, pattern_length)
    
        #Add the padding that corresponds to a cue=0 (that means no replaying yet,
        #but leanrning the input patterns)
        trial = np.vstack((padding_for_trial, trial))
        
        input_trial = np.hstack((trial, null_input))
        
        #Now we can construct the desired ouput. This is basically a "mirrored"
        #version of the input, so construct accordingly: where null_input put
        #the current trial and vice versa. 
        
        #We need no padding for the output (no "cue needed"). Just require 0s
        #when the pattern is being learned. 
        
        
        null_output = np.zeros((1, pattern_length+1))#Add 1 column to have the same length with input
        
        trial = trial[1:,:]
        
        output_trial = np.hstack((null_output, trial))
        
        #Concatanate the generated input/output trials to the the overall 
        #trials array 
        
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
    
    #Use the fact that the positions of interest are the non 0s in
    #the actual array. This is task specific so if the the format
    #of the trials changes then the way to fetch elements also must change!!
    import numpy as np 
    
    indexes_not_zeros = np.where(actual != 0)[0]  
    
    predicted = predicted[indexes_not_zeros]
    actual = actual[indexes_not_zeros]
    
    from sklearn.metrics import mean_squared_error
    
    err = mean_squared_error(actual, predicted)
    
    #Generate a sequence from the same distribution used for the trials
    #This will function as a "null" baseline
    predicted_rand = np.random.uniform(low, high, len(predicted))
    err_null = mean_squared_error(actual, predicted_rand)
    
    return err, err_null, actual, predicted, predicted_rand

#Plot trials for visual inspection of the performance
def plot_trials(trial_nrs, trial_length, actual, predicted, predicted_rand):
    
    import matplotlib.pyplot as plt
    
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
        
        plt.ylabel("values")
        plt.xlabel('trials across time')
        plt.legend(fontsize=("small"), loc=2)


#input_trials_train, output_trials_train = GenerateInputOutputPatterns_Binary(3, 5, 1000)

#input_trials_test, output_trials_test = GenerateInputOutputPatterns_Binary(3, 5, 1000)


pattern_length=10

input_trials_train, output_trials_train = GenerateInputOutputPatterns_Cont(pattern_length=pattern_length, 
                                                                           low=0., 
                                                                           high=1., 
                                                                           nr_of_trials=1000)

input_trials_test, output_trials_test = GenerateInputOutputPatterns_Cont(pattern_length=pattern_length, 
                                                                         low=0., 
                                                                         high=1., 
                                                                         nr_of_trials=1000)



#Biological topology reservoire
from pathlib import Path

import bio2art_import as b2a

#Specify here the folder where your connectomes are contained 
path_to_connectome_folder = Path("/Users/alexandrosgoulas/Data/work-stuff/python-code/Bio2Art/connectomes/")

#Specify here the connectome that you would like to use
file_conn = "C_Marmoset_Normalized.npy"

C, C_Neurons, Region_Neuron_Ids = b2a.bio2art_from_conn_mat(
    path_to_connectome_folder, 
    file_conn, 
    ND=None, 
    SeedNeurons=600, 
    intrinsic_conn=False, 
    target_sparsity=0.1
    )

#What we retouch for the bio2art network is W. To this end, get the unique 
#pair of values that the random reservoir was initialized with and replace
#the actual weights of the C_Neurons

#Get the indexes for the non zero elements of C_Neurons
non_zero_C_Neurons = np.where(C_Neurons != 0)

x_non_zero_C_Neurons = non_zero_C_Neurons[0]
y_non_zero_C_Neurons = non_zero_C_Neurons[1]

rand_indexes_of_non_zeros = np.random.permutation(len(x_non_zero_C_Neurons))

indexes_for_unique1 = int(np.floor(len(rand_indexes_of_non_zeros)/2))

#Assign the same weight as for the random reervoir for a comparison
C_Neurons[(x_non_zero_C_Neurons[rand_indexes_of_non_zeros[0:indexes_for_unique1]], 
            y_non_zero_C_Neurons[rand_indexes_of_non_zeros[0:indexes_for_unique1]])] = .47

C_Neurons[(x_non_zero_C_Neurons[rand_indexes_of_non_zeros[indexes_for_unique1:]], 
            y_non_zero_C_Neurons[rand_indexes_of_non_zeros[indexes_for_unique1:]])] = -.47



#Random reservoire
#Parameters and initialization of the random reservoire
trials_out = 10
n_transient = ((pattern_length*2)+1)*trials_out

n_reservoir = C_Neurons.shape[0]

W = np.random.choice([0, .47, -.47], p=[.8, .1, .1], size=(n_reservoir, n_reservoir))

#Equate density of random and bio topology reservoirs
density_for_reservoir = density_matrix(C_Neurons)
W = threshold_matrix(W, density_for_reservoir)


#Train and test on the random reservoire
esn = ESNPredictive(
    n_inputs=2,
    n_outputs=1,
    n_reservoir=n_reservoir,
    W=W,
    spectral_radius=1.,
    leak_rate=.4,
    n_transient=n_transient,
    teacher_forcing=False,
    regression_params={
        "method": "pinv",
    },
    #random_seed=42
)

esn.fit(input_trials_train, output_trials_train)

prediction_test = esn.predict(input_trials_test)

err_r, err_null_r, actual, predicted, predicted_rand = evaluate_performance(prediction_test, 
                                                                        output_trials_test, 
                                                                        low=0.,
                                                                        high=1.,
                                                                        discard=70)

#Plot some trials
plot_trials([1,10,100], pattern_length, actual, predicted, predicted_rand)

#Train and test on the bio topology reservoire
esn = ESNPredictive(
    n_inputs=2,
    n_outputs=1,
    n_reservoir=n_reservoir,
    W=C_Neurons,
    spectral_radius=1.,
    leak_rate=.4,
    n_transient=n_transient,
    teacher_forcing=False,
    regression_params={
        "method": "pinv",
    },
    #random_seed=42
)

esn.fit(input_trials_train, output_trials_train)

prediction_test = esn.predict(input_trials_test)

err_bio, err_null_bio, actual, predicted, predicted_rand = evaluate_performance(prediction_test, 
                                                                        output_trials_test, 
                                                                        low=0.,
                                                                        high=1.,
                                                                        discard=70)

#Plot some trials
plot_trials([1,10,100], 
            pattern_length, 
            actual,
            predicted,
            predicted_rand)