#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
import numpy as np
import random

from bio2art import utils

# A set of functions for converting biological neuronal networks to artificial
# neuronal networks. The output is a conenctivity matrix (2D numpy array) that 
# can be used in reccurent neuronal networks (e.g., echo state networks).

# Function that simply reads a csv file and returns the matrix that constitutes
# the neuronal network
def from_list(path_to_connectome_folder, data_name):

    """
    Generate matrix W from scv file
    
    Input
    -----
    path_to_connectome_folder: the path to the folder with the csv file
    data_name: the name of the csv file
    
    Output
    ------
    W: the connectivity matrix in the form of a numpy array
    
    """
    
    file_to_open = path_to_connectome_folder / data_name
    
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

# Function that constructs a connectivity matrix network_scaled with the topology 
# that is dictted by biological neuronal networks.     
def from_conn_mat(
        data_name,
        path_to_connectome_folder = None, 
        neuron_density = None, 
        seed_neurons = None, 
        target_sparsity = 0.2,
        intrinsic_conn = True, 
        target_sparsity_intrinsic = 1.,
        intrinsic_wei = 0.8, 
        rand_partition = False,
        keep_diag = True
        ):
    
    """
    Generate matrix network_scaled from a biological connectome
    
    Input
    -----
    
    data_name: string with the name of the connectome file of the connectome you 
        would like to use. Currently available:
        
        'Drosophila'                     49x49 (NxN shape of the npy array)
        'Human_Betzel_Normalized'        57x57 
        'Macaque_Normalized'             29x29
        'Marmoset_Normalized'            55x55
        'Mouse_Gamanut_Normalized'       19x19
        'Mouse_Ypma_Oh'                  56x56
   
    path_to_connectome_folder: The path to the empirical neural network data
        (connectomes). The path must be a passed from the Path subclasss of 
        pathlib Path('path_to_connectome_folder'). 

    neuron_density: numpy array of positive integers with shape N where 
        N network_original.shape[0] with network_original 
        the actual biological connectome (above). Each entry of 
        neuron_density[i] is denoting the number of neurons that we assume 
        to inhabit region i. 
        neuron_density by default gets populated with 1s (1 neuron per region). 
    
        Note that if seed_neurons is not None each entry of neuron_density 
        will be normalized as proportion over the sum(neuron_density)
    
    seed_neurons: Positive integer, default None, denoting the nr of neurons 
        that will be multiplied by neuron_density[i] to result in the number 
        of neurons to be considered for each region i. 
    
        The neuron_density numpy array will be normalized for every region i as:  
        neuron_density[i] / sum(neuron_density)
    
        Note that if seed_neurons=None, the neuron_density array is 
        not scaled and used as is.
    
    target_sparsity: float (0 1], default 0.2, for each source neuron the 
        percentage of all possible neuron-targets to form connections with. 
        Note that at least 1 neuron will function as target in case that the 
        resulting percentage is <1.
        This parameter can be used to make the sparisty of network_scaled vary
        around the density dictated by the actual biological connectomes.
        Note that this parameter is meaningful only if at least one region 
        has more than 1 neuron, that is, for some i, neuron_density[i]>1.
    
    intrinsic_conn: Boolean, default True, denoting if the within regions 
        neuron-to-neuron  connectivity will be generated. 
        
    target_sparsity_intrinsic: float (0 1], default 1., same as 
        target_sparsity, but for the within-region/intrinsic connections
        
    intrinsic_wei: float (0 1], default 0.8, denoting the percentage of the
        weight that will be assigned to the intrinsic weights. 
        E.g., 0.8*sum(extrinsic weight)where sum(extrinsic weight) is the sum 
        of weights of connections from region A to all other regions, but A.
        NOTE: This parameter makes sense only if intrinsic_conn = True. 
    
    rand_partition: Boolean, default False, specifying if the original weight
        of each connection in the empirical connectome will be partitioned
        in k parts that sum to the original connection weight, 
        where k = nr_source_neurons * nr_target_neurons. nr_source_neurons
        and nr_target_neurons are the nr of neurons in the source i and target 
        areas j as part of C_Neurons[i,j]. The original connection weight is
        C[i,j] where C the connectome corresponding to the dataset specified
        on the parameter file.
        If False, then for a given source and target i,j C[i,j] will
        be populated withe equal constant wight values such that:
        C_Neurons[m,n] = C[i,j] / (nr_source_neurons * nr_target_neurons)
        with m, n all index neurons belonging to region i, j respectively. 
    
    keep_diag: Boolean variable, default True denoting if the diagonal entries 
        (denoting self-to-self neuron connections) should be kept of or not. 
        NOTE: This parameter only has an effect when intrinsic_conn = True.
    
    Output
    ------
    network_original: The actual biological neural network that was used in the 
        form of a NxN numpy array (see description of data sets for each N)
    
    network_scaled: the artificial neural network in the form of a 
        numpy array (NxN with N bound to the parameters seed_neurons,
        neuron_density)
    
    region_neuron_ids: A list of list of integers for tracking the neurons of 
        the network_scaled array. region_neuron_ids[1] contains a list with 
        integers that denote the neurons of region 1 in network_scaled as 
        network_scaled[region_neuron_ids[1],region_neuron_ids[1]]
    
    """
    file_conn = 'C_' + data_name + '.npy' # Prefix and suffix for the file
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
    
    # Sum of outgoing weights for each region - used for calculation of 
    # intrinsic weights.
    sum_C_out = np.sum(network_original, 0)
    
    # Initiate the neuron to neuron connectivity matrix
    network_scaled = np.zeros((int(all_neurons), int(all_neurons)))
    
    # The not_zeros index marks the regions with which the current region i
    # is connected to. What needs to be done is conencting the respective 
    # neurons constrained in the regions.
    # We use the region_neuron_ids and the weight value of the region-to-region
    # matrix network_original.
    
    # Start populating by row of the region-to-region matrix network_original
    for i in range(network_original.shape[0]):
        # not-zeros denote the indexes of the areas that are receiving
        # incoming connections from the current region i 
        not_zeros = np.where(network_original[i,:] > 0)[0]
     
        # Get the neuron source indexes
        sources_indexes = region_neuron_ids[i]
        
        if intrinsic_conn is True:
            # Add an intrinsic within-region weight by interconnecting all the 
            # neurons that belong to one region
            
            # Intrinsic weight of within region - default 0.8 
            intrinsic_weight = (intrinsic_wei * sum_C_out[i]) / (1-intrinsic_wei) 
            
            # If target sparsity is specified then calculate the percentage 
            # of intrinsic targets to use (source_indexes)
            if target_sparsity_intrinsic == 1.:
                nr_sources_to_use = len(sources_indexes) 
                # If we do not keep the diagonal then we need nr_sources_to_use - 1
                if keep_diag is False: nr_sources_to_use = nr_sources_to_use - 1
            else:
                # If we do not keep the diagonal, then we have to exlcude
                # one element from the sources_index
                nr_sources_to_use = target_sparsity_intrinsic * len(sources_indexes) 
        
                # Ensure that we keep at least one target neuron
                if nr_sources_to_use < 1:
                    nr_sources_to_use = 1
                else:
                    nr_sources_to_use = int(np.round(nr_sources_to_use)) 
                    
                    # If keep_diag = False and nr_sources_to_use == len(sources_indexes)
                    # make sure that the nr_sources_to_use is 
                    # len(sources_indexes)-1 so that by excluding the self-self
                    # connections (as keep_diag = False dictates) we still
                    # have the proper nr_sources_to_use. 
                    if keep_diag is False and nr_sources_to_use == len(sources_indexes):
                        nr_sources_to_use = nr_sources_to_use - 1 
             
            # Ensure that we keep at least one target neuron in case that
            # neuron_density[i] is 1 and self-self connections are not desired            
            if nr_sources_to_use < 1:
                nr_sources_to_use = 1
                                    
            # Processs the weights of the original neural network based
            # on the rand_partition boolean parameter     
            if rand_partition: 
                partitioned_weights = utils.float_partition(intrinsic_weight, 
                                                            len(sources_indexes) * nr_sources_to_use)
                # Make the idx to use the different partitioned_weights
                # when connecting sources to targets 
                start_partitioned_weights = 0
                stop_partitioned_weights = nr_sources_to_use    
            else:   
                # Normalize weight based on the number of target given the 
                # target_sparsity parameter
                intrinsic_weight = intrinsic_weight / (len(sources_indexes) * nr_sources_to_use)
                       
            # Populate the matrix with broadcasting of indexes
            for sources in sources_indexes:
                # Choose random sources as tagets for each step in the for loop
                # Keep random nr_sources_to_use
                #sources_indexes_to_use = list(range(len(sources_indexes)))
                current_sources_indexes = sources_indexes.copy()
                
                # If keep_diag = False we have to exclude the sources from 
                # the sources_indexes_to_use so that self-self connections
                # do not occur.
                if keep_diag is False:
                    current_sources_indexes.pop(current_sources_indexes.index(sources))
                    
                # Keep random nr_sources_to_use current_sources_indexes    
                random.shuffle(current_sources_indexes)
                current_sources_indexes = current_sources_indexes[:nr_sources_to_use]
                
                if rand_partition:
                    network_scaled[sources, 
                                   current_sources_indexes] = partitioned_weights[start_partitioned_weights:stop_partitioned_weights]
                    # Update idx for using the partitioned_weights
                    start_partitioned_weights = stop_partitioned_weights
                    stop_partitioned_weights = stop_partitioned_weights + nr_sources_to_use
                else:
                    network_scaled[sources, 
                                   current_sources_indexes] = intrinsic_weight
                        
        # Loop through the not zeros indexes and fetch the target neuron ids 
        # that are stored in region_neuron_ids 
        for target in not_zeros:
            target_indexes = region_neuron_ids[target]
            
            # Calculate here the strength of connectivity that should be 
            # assigned to the neuron-to-neuron matrix.
            # The weight is dictated by the number of source and target neurons
            # and the respective region-to-region weight of matrix network_original
            current_weight = network_original[i, target]
            
            # Here we can control the sparsity of connections by choosing
            # the portion of all target_indexes to be used.
            # Hence, apply the target_sparsity parameter if < 1.
            if target_sparsity == 1.:
                nr_targets_to_use = len(target_indexes)
            else:    
                nr_targets_to_use = target_sparsity * len(target_indexes) 
                
                # Ensure that we keep at least one target neuron
                if nr_targets_to_use < 1:
                    nr_targets_to_use = 1
                else:
                    nr_targets_to_use = int(np.round(nr_targets_to_use))                
            
            if rand_partition:
                partitioned_weights = utils.float_partition(current_weight, 
                                                            len(sources_indexes) * nr_targets_to_use)
                # Make the idx to use the different partitioned_weights
                # when connecting sources to targets 
                start_partitioned_weights = 0
                stop_partitioned_weights = nr_targets_to_use    
            else:   
                # Normalize weight based on the number of target given the 
                # target_sparsity parameter
                neuron_to_neuron_weight = current_weight / (len(sources_indexes) * nr_targets_to_use)
                                    
            # Populate the matrix with broadcasting of indexes
            for sources in sources_indexes: 
                # Create random targets (current_target_indexes) for each 
                # source seperately (that is, at each step in this for loop)
                # Keep random N=nr_targets_to_use from the target_indexes 
                random.shuffle(target_indexes)
                current_target_indexes = target_indexes[:nr_targets_to_use]
                
                if rand_partition:
                    network_scaled[sources, current_target_indexes] = partitioned_weights[start_partitioned_weights:stop_partitioned_weights]
                    # Update idx for using the partitioned_weights
                    start_partitioned_weights = stop_partitioned_weights
                    stop_partitioned_weights = stop_partitioned_weights + nr_targets_to_use 
                else:
                    network_scaled[sources, current_target_indexes] = neuron_to_neuron_weight
                      
    return network_original, network_scaled, region_neuron_ids
