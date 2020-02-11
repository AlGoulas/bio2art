#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 09:32:53 2020

@author: alexandrosgoulas
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 14:34:29 2020

@author: alexandrosgoulas
"""

#from matplotlib import pyplot as plt
import numpy as np

import seaborn as sns
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

from random import randint


sns.set(context="notebook", style="whitegrid", font_scale=1.4, 
        rc={'grid.linestyle': '--', 
            'grid.linewidth': 0.8,})

from echoes import ESNPredictive

#Activation functions
def sigmoid(x):
    
   x = 1/(1+np.exp(-x))
     
   return x 

def relu(x):
    
    x = np.maximum(x, 0.)
    
    return x

#Define some usuful functions for matrix density and a custom sigmoid
#to be optionally used as an activation function.

def density_matrix(X):
    
    #import numpy as np
    
    #Calculate the current density of the matrix
    #It included the diagonal!
    X_size = X.shape
    non_zeros = np.where(X != 0)
    
    density = len(non_zeros[0]) / (X_size[0] * X_size[1])
    
    return density


def threshold_matrix(X, desired_density):
    
    #import numpy as np
    
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


def GenerateInputOutputPatterns_Binary(pattern_length=3, nr_of_patterns=5, nr_of_trials=100):
    
    #from random import randint
    #import  numpy as np    
    
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
    
    #import numpy as np
    
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
    #import numpy as np 
    
    indexes_not_zeros = np.where(actual != 0)[0]  
    
    predicted = predicted[indexes_not_zeros]
    actual = actual[indexes_not_zeros]
        
    err = mean_squared_error(actual, predicted)
    
    #Generate a sequence from the same distribution used for the trials
    #This will function as a "null" baseline
    predicted_rand = np.random.uniform(low, high, len(predicted))
    err_null_random = mean_squared_error(actual, predicted_rand)
        
    return err, err_null_random, actual, predicted, predicted_rand


def evaluate_performance_3D(predicted, actual, pattern_length, discard=0):
    
    if discard > 0:
        #Remember that predictions are 3D with:
        #[trials, time_steps(pattern_length), features] 
        actual = actual[discard:, :, :]
        predicted = predicted[discard:, :, :]
    
    #Loop through 1st dim to get trials
    
    all_trials_predicted = None 
    all_trials_actual = None  
        
    for tr in range(0, predicted.shape[0]):
        
        trial_predicted = predicted[tr, (pattern_length+1):, :]
        trial_actual = actual[tr, (pattern_length+1):, :]
        
        if all_trials_predicted is None:
            
            all_trials_predicted = trial_predicted
            all_trials_actual = trial_actual
            
        else:
            
            all_trials_actual = np.hstack((all_trials_actual, trial_actual))  
            all_trials_predicted = np.hstack((all_trials_predicted, trial_predicted)) 
        
    #Calculate the mean square error
    #actual = all_trials    
    err = mean_squared_error(all_trials_actual, all_trials_predicted)    

    actual = all_trials_actual
    predicted = all_trials_predicted
    
    return err, actual, predicted

#Plot trials for visual inspection of the performance
def plot_trials(trial_nrs, trial_length, actual, predicted, predicted_rand, mean_trial_predictor):
    
    #import matplotlib.pyplot as plt
    
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
#is usually the case, that is, "predictions" are jsut the mean of the 
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
    
    for tr in range(0, int(nr_trials_to_consider)):
        
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
    
#Function to convert the trials that are concatanated into a big time 
#series (generated by GenerateInputOutputPatterns_Cont function)
#into a structure that can be fed to a RNN in Keras.
#
#To this end, we need to specify how many trials will constitute a batch
#and convert accordingly the 2D [time_steps, features] (note here time_steps 
#refers to all concatanated trials!) to 
#time series to 3D [samples, time_steps, features] 
        
def Convert_Patterns_to_3D(trials, pattern_length, nr_of_trials):

    #trials_3D must have shape [samples, time_steps, features] 
    #Find the size of the samples given the length of the trials and the 
    #pattern length.
    
    #By construction (see GenerateInputOutputPatterns_Cont) 
    #len(trial) = ((pattern_length*2)+1) * nr_of_trials

    #import numpy as np
    
    time_steps=((pattern_length * 2) + 1)
    
    trials_3D = np.zeros((nr_of_trials, time_steps, trials.shape[1]))
    
    start_index = 0
    stop_index = (pattern_length * 2) + 1
    
    for tr in range(0, nr_of_trials):
        
        trials_3D[tr, :, :] = trials[start_index:stop_index, :]
        
        #Update indexes to grab the next trial
        start_index = start_index + ((pattern_length * 2) + 1)
        stop_index = stop_index + ((pattern_length * 2) + 1)
    
    return trials_3D        


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


#Convert to 3D input appropriate for Keras RNN
input_trials_train_3D = Convert_Patterns_to_3D(input_trials_train, 
                                               pattern_length, 
                                               1000)

output_trials_train_3D = Convert_Patterns_to_3D(output_trials_train, 
                                                pattern_length, 
                                                1000)

input_trials_test_3D = Convert_Patterns_to_3D(input_trials_test, 
                                              pattern_length, 
                                              1000)

#This conversin is NOT necessary, but is better for the time 
#perform it so that we have everything related to the Keras RNN in the 
#ame format (3D) for the sake of categorizing mentaly what is what.

output_trials_test_3D = Convert_Patterns_to_3D(output_trials_test, 
                                              pattern_length, 
                                              1000)

#Determine the low and high limits of a uniform distribution from which the
#weights for the reservoire willl be geenrated

weight_reservoire_high = 1.
weight_reservoire_low = -1.

spectral_radius= .95

leak_rate = 0.4

#Biological topology reservoire
from pathlib import Path

import bio2art_import as b2a

#Specify here the folder where your connectomes are contained 
path_to_connectome_folder = Path("/Users/alexandrosgoulas/Data/work-stuff/python-code/Bio2Art/connectomes/")

#Specify here the connectome that you would like to use
file_conn = "C_Macaque_Normalized.npy"

C, C_Neurons, Region_Neuron_Ids = b2a.bio2art_from_conn_mat(
    path_to_connectome_folder, 
    file_conn, 
    ND=None, 
    SeedNeurons=100, 
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
# C_Neurons[(x_non_zero_C_Neurons[rand_indexes_of_non_zeros[0:indexes_for_unique1]], 
#             y_non_zero_C_Neurons[rand_indexes_of_non_zeros[0:indexes_for_unique1]])] = weight_reservoire_1

# C_Neurons[(x_non_zero_C_Neurons[rand_indexes_of_non_zeros[indexes_for_unique1:]], 
#             y_non_zero_C_Neurons[rand_indexes_of_non_zeros[indexes_for_unique1:]])] = weight_reservoire_2


#Random reservoire
#Parameters and initialization of the random reservoire
trials_out = 20
n_transient = ((pattern_length*2)+1)*trials_out

n_reservoir = C_Neurons.shape[0]

#W = np.random.choice([0, weight_reservoire_1, weight_reservoire_2], p=[.8, .1, .1], size=(n_reservoir, n_reservoir))

W = np.random.uniform(weight_reservoire_low, 
                      weight_reservoire_high, 
                      [n_reservoir, n_reservoir])

#Equate density of random and bio topology reservoirs
density_for_reservoir = density_matrix(C_Neurons)
W = threshold_matrix(W, density_for_reservoir)

#Fill in the weights of the bio reservoire with the exact values used for 
#the random
weights_random_reservoire = W[np.where(W != 0)]
#Assign them to the topology of the bio resevoire
bio_weights_index = np.where(C_Neurons != 0)
C_Neurons[bio_weights_index] = weights_random_reservoire

#Train and test on the random reservoire
esn = ESNPredictive(
    n_inputs=2,
    n_outputs=1,
    n_reservoir=n_reservoir,
    W=W,
    spectral_radius=spectral_radius,
    leak_rate=leak_rate,
    n_transient=n_transient,
    teacher_forcing=False,
    activation=relu,
    regression_params={
        "method": "pinv",
    },
    #random_seed=42
)

esn.fit(input_trials_train, output_trials_train)

prediction_test = esn.predict(input_trials_test)

#No need to generate a seperate err_null_random output. Thus,
#use the same output argument for the random and bio reservoir
#that is: err_null_random, predicted_rand

err_r, err_null_random, actual_r, predicted_r, predicted_rand = evaluate_performance(prediction_test, 
                                                                                     output_trials_test, 
                                                                                     low=0.,
                                                                                     high=1.,
                                                                                     discard=n_transient)

#Compute the error with the null mean trial predictor as well.

#Construct a null mean trial predictor
mean_trial_predictor = get_mean_of_trials(input_trials_train, 
                                           pattern_length, 
                                           discard=n_transient)

#from sklearn.metrics import mean_squared_error
    
err_null_mean = mean_squared_error(actual_r, mean_trial_predictor)



#Plot some trials
plot_trials([1,10,100], 
            pattern_length, 
            actual_r,
            predicted_r,
            predicted_rand,
            mean_trial_predictor)

#Train and test on the bio topology reservoire
esn = ESNPredictive(
    n_inputs=2,
    n_outputs=1,
    n_reservoir=n_reservoir,
    W=C_Neurons,
    spectral_radius=spectral_radius,
    leak_rate=leak_rate,
    n_transient=n_transient,
    teacher_forcing=False,
    activation=relu,
    regression_params={
        "method": "pinv",
    },
    #random_seed=42
)

esn.fit(input_trials_train, output_trials_train)

prediction_test = esn.predict(input_trials_test)

err_bio, err_null_random, actual_bio, predicted_bio, predicted_rand = evaluate_performance(prediction_test, 
                                                                                           output_trials_test, 
                                                                                           low=0.,
                                                                                           high=1.,
                                                                                           discard=n_transient)
#Not needed since it is the same: input trial are the same for r and bio 
#reservoirs
#err_null_mean = mean_squared_error(actual_bio, mean_trial_predictor)

#Plot some trials
plot_trials([1,10,100], 
            pattern_length, 
            actual_bio,
            predicted_bio,
            predicted_rand,
            mean_trial_predictor)

#See how a Simple RNN model from Keras will do
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
 
#Convert the input to 3D for Keras RNN
#from numpy import newaxis
#input_trials_train = input_trials_train[newaxis, :, :]
#output_trials_train = output_trials_train[newaxis, :, :]

model = Sequential()

# model.add(Dense(2,
#                 activation="relu",
#                 input_shape=(None, 2)
#                 )
#           )

model.add(SimpleRNN(10, 
                    input_shape=(None, 2),
                    activation="relu",
                    return_sequences=True
                    )
          )

model.add(Dense(1))

model.summary()

batch_size = 10
#batch_size = batch_size * ((pattern_length*2)+1) * batch_size 

model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mae'])
history = model.fit(input_trials_train_3D, 
                    output_trials_train_3D,
                    epochs=20,
                    batch_size=batch_size,
                    validation_split=0.2,
                    shuffle=False)


loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#Make predictions on test trials
prediction_RNN = model.predict(input_trials_test_3D, 
                              batch_size=batch_size,
                              verbose=1)

#Evaluate the performance
err_keras_rnn, actual_keras_rnn, predicted_keras_rnn = evaluate_performance_3D(prediction_RNN, 
                                                                               output_trials_test_3D,
                                                                               pattern_length,
                                                                               trials_out)

predicted_keras_rnn = np.reshape(predicted_keras_rnn, 
                                 (predicted_keras_rnn.shape[0] * predicted_keras_rnn.shape[1], 1), 
                                 order='F')

actual_keras_rnn = np.reshape(actual_keras_rnn, 
                                 (actual_keras_rnn.shape[0] * actual_keras_rnn.shape[1], 1), 
                                 order='F')

plot_trials([1,10,100], 
            pattern_length, 
            actual_keras_rnn,
            predicted_keras_rnn,
            predicted_rand,
            mean_trial_predictor)