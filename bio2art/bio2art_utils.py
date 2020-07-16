#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle 

# Return the names of the regions of the indicated connectome
def get_names(path_to_connectome_folder, data_name):
    '''
    Input
    -----
    path_to_connectome_folder: the path to the folder with connectome files
    
    data_name: string with the name of the connectome file of the connectome you 
    would like to use. Currently available:
    Drosophila                     
    Human_Betzel_Normalized        
    Macaque_Normalized             
    Marmoset_Normalized            
    Mouse_Gamanut_Normalized       
    Mouse_Ypma_Oh
    
    Output
    ------
    names: list of strings of len N, with N the number of areas in the 
    connectome, indicating the names of each brain region
    
    Note: Pickle is used for loading the names
    
    '''
    data_name = 'Names_' + data_name + '.lst' # Prefix and suffix for the file
    file_to_open = path_to_connectome_folder / data_name
    with open(file_to_open, 'rb') as fp: 
        names = pickle.load(fp)
    
    return names 
