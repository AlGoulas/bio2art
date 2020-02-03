#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 13:31:03 2019

@author: alexandrosgoulas
"""


def bio2art_from_list(path_to_connectome_folder, file):

    import numpy as np
    import csv
    
    file_to_open = path_to_connectome_folder / file
    
    #Lists to save the name of the neurons
    #It will be needed to convert the csv fiel to an adjacency matrix
    
    #from_list = []
    #to_list = []
    
    all_neuron_names = []
    
    from_indexes_list =[]
    to_indexes_list =[]
    value_connection_list = []
    
    with open(file_to_open, newline='') as f:
        reader = csv.reader(f)
        
        for row in reader:
            
            #Check if the row contains all data
            index = [i for i, list_item in enumerate(row) if list_item == ""]
            
            #If row contains all data, then proceed
            if len(index) == 0:
            
                from_neuron = row[0]  
                to_neuron = row[1]
                
                #strip the strings from spaces so we do not create duplicates
                from_neuron = from_neuron.strip()
                to_neuron = to_neuron.strip()
                
                #Keep track of all the neuron names in the 
                index_from = [i for i, list_item in enumerate(all_neuron_names) if list_item == from_neuron]
                
                if len(index_from) > 0:
                    
                    from_indexes_list.append(index_from[0])
                    
                else:
                    #If it is not in the from neuron list, added and make the index the 
                    #len of list AFTER we add the new name
                    all_neuron_names.append(from_neuron)
                    from_indexes_list.append(len(all_neuron_names)-1)
                  
                #Do the same for the to_neuron     
                index_to = [i for i, list_item in enumerate(all_neuron_names) if list_item == to_neuron]
                
                if len(index_to) > 0:
                    
                    to_indexes_list.append(index_to[0])
                    
                else:
                    #If it is not in the from neuron list, added and make the index the 
                    #len of list AFTER we add the new name
                    all_neuron_names.append(to_neuron)
                    to_indexes_list.append(len(all_neuron_names)-1)    
                    
                
                #Irrespective of the above conditions the value of the connection
                #is stored in its respective list
                value_connection_list.append(float(row[len(row)-2]))
                
    #Build the connectivity matrix
    W = np.zeros((len(all_neuron_names), len(all_neuron_names))) 
    
    for i in range(len(to_indexes_list)-1):
        W[from_indexes_list[i]][to_indexes_list[i]] = value_connection_list[i]
        
    return W        


def bio2art_from_conn_mat(path_to_connectome_folder, file_conn, ND=None, SeedNeurons=10, intrinsic_conn=True, target_sparsity=0.2, intrinsic_wei=0.8):
    
    import numpy as np
    #import csv
    
    file_to_open = path_to_connectome_folder / file_conn
    
    #Read the connectivity matrix - it must be stored as a numpy array
    C = np.load(file_to_open)
    
    #file_to_open = path_to_connectome_folder / file_ND
    
    #ND = np.load(file_to_open)
    
    #What needs to be done is:
    #Use the ND vector to create and connect regions containing neurons that 
    #contain SeedNeurons*ND[i] where ND is the vector specifying the percentage 
    #of neurons for each region i.
    
    if(ND==None):
        ND=np.ones((C.shape[0],1))
    
    sum_ND = np.sum(ND)
    ND_scaled_sum = ND / sum_ND
    
    #This is how many neurons each region should have
    Nr_Neurons = np.round(ND_scaled_sum * SeedNeurons)
    
    #Construct the neuron to neuron matrix - it is simply an array of unique
    #integer ids of all the neurons dictated by sum(Nr_Neurons)
    
    all_neurons = np.sum(Nr_Neurons)
    
    index_neurons = [i for i in range(int(all_neurons))]
    index_neurons = np.asarray(index_neurons)
    #index_neurons = index_neurons + 1
    
    #Create a list of lists that tracks the neuron ids that each region 
    #contains
    
    Region_Neuron_Ids=[]
    start = 0
    
    for i in range(C.shape[0]):
        offset = Nr_Neurons[i]
        offset = int(offset)
        
        new_list_of_region = list(range(start, (start + offset)))
        Region_Neuron_Ids.append(new_list_of_region)
        
        #Update the indexes
        start = start + offset
    
    #Rescale the weights so that the outgoing strength of each regions
    #is equal to 1
    sum_C_out = np.sum(C, 0)
    #C_Norm = C / sum_C_out
    
    #Initiate the neuron to neuron connectivity matrix
    C_Neurons = np.zeros((int(all_neurons), int(all_neurons)))
    
    #The not_zeros index marks the regions with which the current region i
    #is connected to. What needs to be done is conencting the respective 
    #neurons constrained in the regions.
    #We use the Region_Neuron_Ids and the weight value of the region-to-region
    #matrix C.
    
    #Start populating by row of the region-to region matrix C
    for i in range(C.shape[0]):
        
        #not-zeros denote the indexes of the areas that are receiving
        #incoming connections from the current region i 
        not_zeros = np.where(C[i,:] > 0)[0]
        #not_zeros = not_zeros[0]
        #Get the neuron source indexes
        sources_indexes = Region_Neuron_Ids[i]
        
        if(intrinsic_conn == True):
            #Add an intrinsic within region weight by interconencting all the 
            #neurons that belong to one region
            
            #Intrinsic weight of within region - default 80%
            intrinsic_weight = (intrinsic_wei * sum_C_out[i]) / (1-intrinsic_wei) 
            
            #Populate the matrix with broadcasting of indexes
            for sources in range(len(sources_indexes)):
                C_Neurons[sources_indexes[sources], sources_indexes] = intrinsic_weight
            
            
        #Loop through the not zeros indexes and fetch the target neuron 
        #Ids that are stored in Region_Neuron_Ids
        for target in range(len(not_zeros)):
            target_indexes = Region_Neuron_Ids[not_zeros[target]]
            
            #Calculate here the strength of connectivity that should be 
            #assigned to the neuron-to-neuron matrix.
            #
            #The weight is dictated by the number of source and target neurons
            #and the respective region-to-region weight of matrix C
            
            current_weight = C[i, not_zeros[target]]
            
            neuron_to_neuron_weight = current_weight / (len(sources_indexes)*len(target_indexes))
            
            #For now the neuron-to-neuron weight is identical due to 
            #lack of the precise number from experimental observations.
            #It migh be needed to inject soem noise for variations to emerge.
            
            #Populate the matrix with broadcasting of indexes
            for sources in range(len(sources_indexes)):
                #C_Neurons[sources_indexes[sources], target_indexes] = neuron_to_neuron_weight
                
                #Here we can control the sparsity of connections by choosing
                #the portion of all target_indexes to be used.
                #Hence, apply the target_sparsity parameter
                nr_targets_to_use = target_sparsity*len(target_indexes)
                
                #Ensure that we keep at least one target neuron
                if nr_targets_to_use < 1:
                    nr_targets_to_use = 1
                else:
                    nr_targets_to_use = int(np.round(nr_targets_to_use))
                
                #Keep random nr_targets_to_use
                target_indexes_to_use = np.random.permutation(len(target_indexes))
                target_indexes_to_use = target_indexes_to_use[0:nr_targets_to_use]
                target_indexes = np.asarray(target_indexes)[target_indexes_to_use]            
                
                C_Neurons[sources_indexes[sources], target_indexes] = neuron_to_neuron_weight
                
            #Remove self-to-self strength/connections
            #Maybe in the future this can be parametrized as a desired 
            #feature to be included or not    
            np.fill_diagonal(C_Neurons, 0.)    
                
    
    return C, C_Neurons, Region_Neuron_Ids


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
    

#Convert connectome to matrix     
from pathlib import Path

#path_to_connectome_folder = Path("/Users/alexandrosgoulas/Data/work-stuff/python-code/Bio2Art/connectomes/c.elegans")
#file = "male_full_edgelist.csv"            

#W = bio2art_from_list(path_to_connectome_folder, file)   

path_to_connectome_folder = Path("/Users/alexandrosgoulas/Data/work-stuff/python-code/Bio2Art/connectomes/")
file_conn = "C_Marmoset_Normalized.npy"


C, C_Neurons, Region_Neuron_Ids = bio2art_from_conn_mat(path_to_connectome_folder, file_conn, None, 600, False, target_sparsity=0.1)

#Keep in this variable the size of the C_Neurons to intiialize the reservoir
size_of_matrix = C_Neurons.shape[0]


#Echo state network memory

import numpy as np

from echoes.tasks import MemoryCapacity
from echoes.plotting import plot_forgetting_curve, set_mystyle
set_mystyle() # make nicer plots, can be removed

# Echo state network parameters (after Jaeger)
n_reservoir = size_of_matrix
W = np.random.choice([0, .47, -.47], p=[.6, .2, .2], size=(size_of_matrix, size_of_matrix))
W_in = np.random.choice([.2, -.2], p=[.5, .5], size=(n_reservoir, 2))



density_for_reservoir = density_matrix(C_Neurons)
W = threshold_matrix(W, density_for_reservoir)

# Task parameters (after Jaeger)
inputs_func=np.random.uniform
inputs_params={"low":-.1, "high":.1, "size":300}
lags = [1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 100, 150, 200]

esn_params = dict(
    n_inputs=1,
    n_outputs=len(lags),  # automatically decided based on lags
    n_reservoir=size_of_matrix,
    W=W,
    W_in=W_in,
    spectral_radius=1,
    bias=0,
    n_transient=100,
    regression_params={
        "method": "pinv"
    },
    random_seed=42,
)

# Initialize the task object
mc = MemoryCapacity(
    inputs_func=inputs_func,
    inputs_params=inputs_params,
    esn_params=esn_params,
    lags=lags
).fit_predict()  # Run the task

plot_forgetting_curve(mc.lags, mc.forgetting_curve_)


#What we retouch for the bio2art network is W. To this end, get the unique 
#pair of values that the random reservoir was initialized with and replace
#the actual weights of the C_Neurons

#Get the indexes for the non zero elements of C_Neurons
non_zero_C_Neurons = np.where(C_Neurons != 0)

x_non_zero_C_Neurons = non_zero_C_Neurons[0]
y_non_zero_C_Neurons = non_zero_C_Neurons[1]

rand_indexes_of_non_zeros = np.random.permutation(len(x_non_zero_C_Neurons))

indexes_for_unique1 = int(np.floor(len(rand_indexes_of_non_zeros)/2))

C_Neurons[(x_non_zero_C_Neurons[rand_indexes_of_non_zeros[0:indexes_for_unique1]], 
            y_non_zero_C_Neurons[rand_indexes_of_non_zeros[0:indexes_for_unique1]])] = .47

C_Neurons[(x_non_zero_C_Neurons[rand_indexes_of_non_zeros[indexes_for_unique1:]], 
            y_non_zero_C_Neurons[rand_indexes_of_non_zeros[indexes_for_unique1:]])] = -.47

esn_params = dict(
    n_inputs=1,
    n_outputs=len(lags),  # automatically decided based on lags
    n_reservoir=size_of_matrix,
    W=C_Neurons,
    W_in=W_in,
    spectral_radius=1,
    bias=0,
    n_transient=100,
    regression_params={
        "method": "pinv"
    },
    random_seed=42,
)

# Initialize the task object
mc_bio = MemoryCapacity(
    inputs_func=inputs_func,
    inputs_params=inputs_params,
    esn_params=esn_params,
    lags=lags
).fit_predict()  # Run the task

plot_forgetting_curve(mc_bio.lags, mc_bio.forgetting_curve_)
      