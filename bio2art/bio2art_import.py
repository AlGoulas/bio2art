#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import csv

# A set of functions for converting biological neuronal networks to artificial
# neuronal networks. The output is a conenctivity matrix (2D numpy array) that 
# can be used in reccurent neuronal networks (e.g., echo state networks).

# Function that simply reads a csv file and returns the matrix that constitutes
# the neuronal network
def bio2art_from_list(path_to_connectome_folder, file):

    """
    Generate matrix W from scv file
    
    input:
    path_to_connectome_folder: the path to the folder with the csv file
    file: the name of the csv file
    
    output:
    W: the connectivity matrix in the form of a numpy array
    
    """
    
    file_to_open = path_to_connectome_folder / file
    
    # Lists to save the name of the neurons
    # It will be needed to convert the csv fiel to an adjacency matrix    
    all_neuron_names = []
    
    from_indexes_list =[]
    to_indexes_list =[]
    value_connection_list = []
    
    with open(file_to_open, newline='') as f:
        reader = csv.reader(f)
        
        for row in reader:
            
            # Check if the row contains all data
            index = [i for i, list_item in enumerate(row) if list_item == ""]
            
            # If row contains all data, then proceed
            if len(index) == 0:
            
                from_neuron = row[0]  
                to_neuron = row[1]
                
                # Strip the strings from spaces so we do not create duplicates
                from_neuron = from_neuron.strip()
                to_neuron = to_neuron.strip()
                
                # Keep track of all the neuron names in the 
                index_from = [i for i, list_item in enumerate(all_neuron_names) if list_item == from_neuron]
                
                if len(index_from) > 0:
                    
                    from_indexes_list.append(index_from[0])
                    
                else:
                    # If it is not in the from neuron list, added and make the index the 
                    # len of list AFTER we add the new name
                    all_neuron_names.append(from_neuron)
                    from_indexes_list.append(len(all_neuron_names)-1)
                  
                # Do the same for the to_neuron     
                index_to = [i for i, list_item in enumerate(all_neuron_names) if list_item == to_neuron]
                
                if len(index_to) > 0:
                    
                    to_indexes_list.append(index_to[0])
                    
                else:
                    #If it is not in the from neuron list, added and make the index the 
                    #len of list AFTER we add the new name
                    all_neuron_names.append(to_neuron)
                    to_indexes_list.append(len(all_neuron_names)-1)    
                    
                
                # Irrespective of the above conditions the value of the connection
                # is stored in its respective list
                value_connection_list.append(float(row[len(row)-2]))
                
    # Build the connectivity matrix
    W = np.zeros((len(all_neuron_names), len(all_neuron_names))) 
    
    for i in range(len(to_indexes_list)-1):
        W[from_indexes_list[i]][to_indexes_list[i]] = value_connection_list[i]
        
    return W        

# Function that constructs a conenctivity matrix network_scaled with the topology 
# that is dictted by biological neuronal networks.     
def bio2art_from_conn_mat(
        path_to_connectome_folder, 
        file_conn, 
        neuron_density=None, 
        seed_neurons=None, 
        intrinsic_conn=True, 
        target_sparsity=0.2, 
        intrinsic_wei=0.8,
        keep_diag=True
        ):
    
    """
    Generate matrix network_scaled from a biological connectome
    
    input:
    path_to_connectome_folder: the path to the folder with connectome files
    
    file: string with the name of the connectome file of the connectome you 
    would like to use. Currently available:
    C_Drosophila.npy                 49x49 (shape of the npy array)
    C_Human_Betzel_Normalized.npy    57x57 
    C_Macaque_Normalized.npy         29x29
    C_Marmoset_Normalized.npy        55x55
    C_Mouse_Gamanut_Normalized.npy   19x19
    C_Mouse_Ypma_Oh.npy              56x56
    
    neuron_density: numpy array of positive integers withshape N where 
    N network_original.shape[0] with network_original 
    the actual biological connectome (above). Each entry of neuron_density[i] 
    is denoting the number of neurons that we assume to inhabit region i. 
    neuron_density by default gets populated with 1s. Note that if seed_neurons 
    is not None each entry of neuron_density will be normalized as proportion 
    over the sum(neuron_density)
    
    seed_neurons: Positive integer denoting the nr of neurons that will be 
    multiplied by neuron_density[i] to result in the number of neurons to be 
    considered for each region i. 
    Default None. Note that if seed_neurons=None, the neuron_density array is 
    not scaled and used as is.
    
    intrinsic_conn: Boolean denoting if the within regions neuron-to-neuron
    conenctivity will be generated. Note that currently all-to-all
    within region conenctions are assumed and implemented. Default True. 
    
    target_sparsity: float (0 1] for each source neuron the percentage of all 
    possible neuron-targets to form connections with. Note that at least 1 
    neuron will function as target in case that the resulting percentage is <1.
    This parameter can be used to make the sparisty of network_scaled vary
    around the density dictated by the actual biological connectomes.
    Default=0.2. 
    
    intrinsic_wei: float (0 1] denoting the percentage of the weight that 
    will be assigned to the intrinsic weights. E.g., 0.8*sum(extrinsic weight)
    where sum(extrinsic weight) is the sum of weights of connections from 
    region A to all other regions, but A.
    
    keep_diag: Boolean variable denoting if the diagonal entries (denoting 
    self-to-self neuron connections) should be kept of or not. Default True. 
    Note that this parameter only has an effect  when intrinsic_conn is True.
    
    output:
    network_original: The actual biological connectome that was used in the 
    form of a numpy array
    
    network_scaled: the artificial neuronal network in the form of a 
    numpy array
    
    region_neuron_ids: A list of list of integers for tracking the neurons of 
    the network_scaled array. region_neuron_ids[1] contains a list with 
    integers that denote the neurons of region 1 in network_scaled as 
    network_scaled[region_neuron_ids[1],region_neuron_ids[1]]
    
    """
    
    file_to_open = path_to_connectome_folder / file_conn
    
    # Read the connectivity matrix - it must be stored as a numpy array
    network_original = np.load(file_to_open)
    
    # What needs to be done is:
    # Use the neuron_density vector to create and connect regions containing neurons that 
    # contain seed_neurons*neuron_density[i] where neuron_density is the vector specifying the 
    # percentage of neurons for each region i.
    if neuron_density is None:
        neuron_density=np.ones((network_original.shape[0],))
          
    if(neuron_density.shape[0] != network_original.shape[0]):
        print("Size of neuron_density must be equal to value of connectome:", network_original.shape[0])
        return 
    
    # If seed_neurons is specified, then we scale the neuron_density so that each entry
    # neuron_density[i] is a percentage over all neurons contained in neuron_density. Then each entry 
    # is scaled up by getign multiplied by the seed_neurons. 
    if seed_neurons is not None:    
        sum_ND = np.sum(neuron_density)
        ND_scaled_sum = neuron_density / sum_ND
        Nr_Neurons = np.ceil(ND_scaled_sum * seed_neurons)
        all_neurons = np.sum(Nr_Neurons)
        
    # If no seed_neurons is used, then the all_neurons variable is simply 
    # sum(neuron_density) and the Nr_Neurons to work with is siply the unscaled
    # neuron_density array as passed as an input argument.    
    if seed_neurons is None:
       all_neurons = np.sum(neuron_density)
       Nr_Neurons = neuron_density
    
    # Construct the neuron to neuron matrix - it is simply an array of unique
    # integer ids of all the neurons dictated by all_neurons  
    index_neurons = [i for i in range(int(all_neurons))]
    index_neurons = np.asarray(index_neurons)

    # Create a list of lists that tracks the neuron ids that each region 
    # contains
    region_neuron_ids=[]
    start = 0
    
    for i in range(network_original.shape[0]):
        offset = Nr_Neurons[i]
        offset = int(offset)
        
        new_list_of_region = list(range(start, (start + offset)))
        region_neuron_ids.append(new_list_of_region)
        
        #Update the indexes
        start = start + offset
    
    # Rescale the weights so that the outgoing strength of each regions
    # is equal to 1
    sum_C_out = np.sum(network_original, 0)
    
    # Initiate the neuron to neuron connectivity matrix
    network_scaled = np.zeros((int(all_neurons), int(all_neurons)))
    
    # The not_zeros index marks the regions with which the current region i
    # is connected to. What needs to be done is conencting the respective 
    # neurons constrained in the regions.
    # We use the region_neuron_ids and the weight value of the region-to-region
    # matrix network_original.
    
    # Start populating by row of the region-to region matrix network_original
    for i in range(network_original.shape[0]):
        
        # not-zeros denote the indexes of the areas that are receiving
        # incoming connections from the current region i 
        not_zeros = np.where(network_original[i,:] > 0)[0]
     
        # Get the neuron source indexes
        sources_indexes = region_neuron_ids[i]
        
        if intrinsic_conn is True:
            # Add an intrinsic within region weight by interconencting all the 
            # neurons that belong to one region
            
            # Intrinsic weight of within region - default 80%
            intrinsic_weight = (intrinsic_wei * sum_C_out[i]) / (1-intrinsic_wei) 
            
            # Populate the matrix with broadcasting of indexes
            for sources in range(len(sources_indexes)):
                network_scaled[sources_indexes[sources], sources_indexes] = intrinsic_weight
            
            
        # Loop through the not zeros indexes and fetch the target neuron 
        # Ids that are stored in region_neuron_ids
        for target in range(len(not_zeros)):
            target_indexes = region_neuron_ids[not_zeros[target]]
            
            # Calculate here the strength of connectivity that should be 
            # assigned to the neuron-to-neuron matrix.
            # The weight is dictated by the number of source and target neurons
            # and the respective region-to-region weight of matrix network_original
            current_weight = network_original[i, not_zeros[target]]
            
            neuron_to_neuron_weight = current_weight / (len(sources_indexes)*len(target_indexes))
            
            # For now the neuron-to-neuron weight is identical due to 
            # lack of the precise number from experimental observations.
            # It migh be needed to inject soem noise for variations to emerge.
            
            # Populate the matrix with broadcasting of indexes
            for sources in range(len(sources_indexes)):
                
                # Here we can control the sparsity of connections by choosing
                # the portion of all target_indexes to be used.
                # Hence, apply the target_sparsity parameter
                nr_targets_to_use = target_sparsity*len(target_indexes)
                
                # Ensure that we keep at least one target neuron
                if nr_targets_to_use < 1:
                    nr_targets_to_use = 1
                else:
                    nr_targets_to_use = int(np.round(nr_targets_to_use))
                
                # Keep random nr_targets_to_use
                target_indexes_to_use = np.random.permutation(len(target_indexes))
                target_indexes_to_use = target_indexes_to_use[0:nr_targets_to_use]
                target_indexes = np.asarray(target_indexes)[target_indexes_to_use]            
                
                network_scaled[sources_indexes[sources], target_indexes] = neuron_to_neuron_weight
                
    # Remove self-to-self strength/connections (diagonal of network_scaled) 
    if keep_diag is False:    
        np.fill_diagonal(network_scaled, 0.)    
                   
    return network_original, network_scaled, region_neuron_ids


from pathlib import Path
path_to_connectome_folder = Path("/Users/alexandrosgoulas/Data/work-stuff/python-code/development/Bio2Art/connectomes/")

file_conn = "C_Macaque_Normalized.npy"# the macaque monkey neuronal network (see bio2art_from_conn_mat for all names of available connectomes)

neuron_density=np.zeros(29,)
neuron_density[:] = 10

network_original, network_scaled, region_neuron_ids = bio2art_from_conn_mat(
    path_to_connectome_folder, 
    file_conn, 
    neuron_density=None, 
    seed_neurons=None, 
    intrinsic_conn=True, 
    target_sparsity=0.1,
    keep_diag=True
    )

