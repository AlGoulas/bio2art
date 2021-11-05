#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle 
import numpy as np
from pathlib import Path
import pkg_resources
import random

# Load the names of the regions of the indicated connectome
def get_names(data_name, path_to_connectome_folder = None):
    '''
    Load the names of the regions of the indicated dataset
    
    Input
    -----
    data_name: str 
        String denoting the name of the neuronal network would like to use. 
        Currently available:
        
        'Drosophila'                     49x49 (NxN shape of the ndarray)
        'Human_Betzel_Normalized'        57x57 
        'Macaque_Normalized'             29x29
        'Marmoset_Normalized'            55x55
        'Mouse_Gamanut_Normalized'       19x19
        'Mouse_Ypma_Oh'                  56x56
        
    path_to_connectome_folder: (optional), default None, object of class pathlib.PosixPath 
        The path to the empirical neural network data (connectomes). 
        The path must be a passed from the Path subclasss of 
        pathlib: path_to_connectome_folder = Path('path_to_desired_dataset'). 
        If not specified, the path to the packaged data will be used.
     
    Output
    ------
    names: list of str 
        list has len N, with N the number of areas in the 
        connectome, indicating the names of each brain region/node
    
    Note: Pickle is used for loading the names
    
    '''
    if path_to_connectome_folder is None:
        path_to_connectome_folder = pkg_resources.resource_filename('bio2art', 'connectomes/')
        path_to_connectome_folder = Path(path_to_connectome_folder)
    data_name = 'Names_' + data_name + '.lst' # Prefix and suffix for the file
    file_to_open = path_to_connectome_folder / data_name
    with open(file_to_open, 'rb') as fp: 
        names = pickle.load(fp)
    
    return names
 
#  Load the neuron density of the regions of the indicated dataset
def get_neuron_density(data_name, path_to_connectome_folder = None):
    '''
    Load the neuron density of the regions of the indicated dataset
    
    Input
    -----
    data_name: str 
        String denoting the name of the neuronal network would like to use. 
        Currently available:
                  
        'Macaque_Normalized'             
        'Marmoset_Normalized'  
        
    path_to_connectome_folder: (optional), default None, object of class pathlib.PosixPath 
        The path to the empirical neural network data (connectomes). 
        The path must be a passed from the Path subclasss of 
        pathlib: path_to_connectome_folder = Path('path_to_desired_dataset'). 
        If not specified, the path to the packaged data will be used.          

    Output
    ------
    neuron_density: ndarray of int of shape (N,)
        N is the number of nodes in the neural networks. Each entry of 
        neuron_density denotes the neuron density (nr of neurons per mm3) 
        for each region/node.
    
    '''
    if path_to_connectome_folder is None:
        path_to_connectome_folder = pkg_resources.resource_filename('bio2art', 'connectomes/')
        path_to_connectome_folder = Path(path_to_connectome_folder)
    file_conn = 'ND_' + data_name + '.npy' # Prefix and suffix for the file
    file_to_open = path_to_connectome_folder / file_conn
    neuron_density = np.load(file_to_open)
    
    return neuron_density.astype('int64') 

#  Construct a scaled neuron_density array based on the seed_neuron
def scale_neuron_density(neuron_density, 
                         seed_neuron = 1,
                         scale_type = 'rank'):
    '''
    Construct a scaled neuron_density ndarray based on rank ordered or ratios 
    of neuron_density values and seed_neuron 
     
    Input
    -----
    neuron_density: ndarray of int of shape (N,)
        N the number of areas in the connectome with entry i denoting the 
        neuron density of region i 
        (returned from function get_neuron_density).
    
    seed_neuron: int, default 1
        specifying the number that will be multiplied by the scaled 
        neuron_density.
        
    scale_type: str 'ratio' 'rank', default 'rank'
        Specifying how the neuron_density values will be scaled. 
        'rank': the values are rank ordered and multiplied by seed_neuron
        'ratio': the values are converted to ratios, 
        neuron_density[i] / min(neuron_density) and multiplied by seed_neuron 
       
    Output
    ------
    scaled_neuron_density: ndarray of int of shape (N,)
        N the number of nodes in the neuronal network, indicating the scaled 
        neuron_density of each brain region/node
    
    '''
    # Copy and sort the neuron densities
    neuron_density_srt = neuron_density.copy()
    neuron_density_srt.sort()
    
    # Get the idx so you can go match neuron_density_srt to neuron_density 
    # The y_ind will contain the desired idx such that:
    # neuron_density[y_ind[i]] == nd_sorted[i] for every i=0,1,2...N 
    # with N=neuron_density.shape[0] 
    xy, x_ind, y_ind = np.intersect1d(neuron_density_srt, 
                                      neuron_density, 
                                      return_indices=True)
    
    scaled_neuron_density = neuron_density.copy()
    
    if scale_type == 'rank':
        scaled_neuron_density[y_ind] = np.asarray(range(1, neuron_density.shape[0]+1)
                                                 ) * seed_neuron
    elif scale_type == 'ratio':
        for i, item in enumerate(neuron_density_srt):
            if i == 0: 
                scaled_neuron_density[y_ind[i]] = 1
                denominator = neuron_density_srt[i]
            scaled_neuron_density[y_ind[i]] = round((item / denominator) * seed_neuron)       
                                   
    return scaled_neuron_density.astype('int64')       

# Partition integer i in n random integers that sum to i 
def _int_partition(i, n):
    '''
    Partition integer i in n random integers that sum to i 
    
    Input
    -----
    i:  int
        a positive integer that needs to be partitioned in n integers
        that sum to i 
    n:  int
        a positive integer specifying the number of partitions/integers
        to be generated
         
    Output
    ------   
    partitions, list of int
        list of n integers with the property 
        sum(partitions)==i 
    '''
    partitions=[]
    if n > i:
        print('\nInteger i must be higher or equal to n\n') 
        return
    if i < 0 or n < 0:
        print('\nIntegers i and n must be positive\n') 
        return
             
    spectrum = (i-n)+2
    for k in range(n-1):
        if spectrum > 1:
            new_val = np.random.randint(1, high=spectrum)
            partitions.append(new_val)
            spectrum = spectrum-new_val
        else:
            new_val = np.random.randint(1, high=2)
            partitions.append(new_val)
   
    partitions.append(i-sum(partitions))
    # Shuffle list so that the order of generation does not bias magnitude 
    # of integers as they are generated
    random.shuffle(partitions)
    
    return partitions

# Partition float f in n random floats that sum to f 
def _float_partition(f, n):
    '''
    Partition float f in n random floats that sum to f 
    
    Input
    -----
    f:  float
        a positive float that needs to be partitioned in n floats
        that sum to i 
    n:  int
        a positive integer specifying the number of partitions/integers
        to be generated
         
    Output
    ------   
    partitions, list of float
        list of n float numbers with the property 
        sum(partitions)==f
    '''
    partitions=[]             
    spectrum = f
    for k in range(n-1):
        new_val = random.uniform(0., abs(spectrum))
        partitions.append(new_val)
        spectrum = spectrum-new_val
   
    partitions.append(f-sum(partitions))
    # Shuffle list so that the order of generation does not bias magnitude 
    # of integers as they are generated
    random.shuffle(partitions)
    
    return partitions            