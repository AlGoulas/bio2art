# Bio2Art
Convert biological neural networks to recurrent neural networks based on the topology dictated by the empirical biological networks.

![bio_and_art_connectomes](bio_and_art_connectomes.png)

# Description

The Bio2Art offers an easy to use function to convert biological neural networks to artificial recurrent neural networks. To this end, empirical neural networks of diverse species are used. Currently, the neural networks of the following species can be used:

1. Macaque monkey (Macaca mulatta)
2. Marmoset monkey (Callithrix jacchus)
3. Mouse (Mus musculus)
4. Human (Homo sapiens)
5. Fly (Drosophila melanogaster)

Note that the term "connectome" refers to a biological neural network.

Bio2Art builds artifical recurrent neural networks by using the topology dictated by the aforementioned empirical neural networks and by extrapolating from the empirical data to scale up the artifical neural networks. 

For instance, if the empirical data correspond to a neural network involving 29 brain regions, then the resulting artificial recurrent neural network can be scaled up by assuming a certain number of neurons populating each region (see examples below and documentation of the importnet.py function). Thus, the output can be an artificial recurrent neural network with an arbitrary number of neurons (e.g., >>29 brain regions), but, importantly, this networks obeys the topology of a desired biological neural network (or "connectome").   

The constructed artificial recurrent neural network is returned as a ndarray and, thus, can be used with virtually any type of artifical recurrent network, for instance, echo state networks.  

# Installation

Download or clone the repository. Open a terminal and change to the corresponding folder. Type:

```
pip install .
```
Note that the Bio2Art only uses numpy (tested with numpy==1.16.2). However, to use the examples (see below), further libraries are needed. Therefore, for executing the examples, create a virtual environment (e.g., with conda) with the requirements enlisted in the requirements.txt file in the "examples" folder.  

# Examples
# Basic use
Please see the documentation of the importnet.from_conn_mat function for a detailed description of the parameters used below. The use of the parameters and their impact is highlighted in the following examples.  

Converting the macaque monkey neural network to a recurrent artifical neural network.

```
from bio2art import importnet
from pathlib import Path

# path to where the "connectomes" folder is located (it is included with the current repository)
path_to_connectome_folder = Path("/.../Bio2Art/connectomes/")#change to the folder where the desired data are located (data that come with this repository)

data_name = "Macaque_Normalized"# the macaque monkey neuronal network (see importnet.from_conn_mat function for all names of available connectomes)

net_orig, net_scaled, region_neuron_ids = importnet.from_conn_mat(
	 data_name=data_name, 
    path_to_connectome_folder=path_to_connectome_folder,  
    neuron_density=None, 
    seed_neurons=None, 
    intrinsic_conn=False, 
    target_sparsity=0.1
    )
```
The neuron_density is the recurrent neural network based on the indicated empirical monkey neuronal network. However, since neuron_density=None and seed_neurons=None, net_scaled is exactly the same with net_orig, that is, the exact same empirical monkey neural network. Not very useful. Let's see how we can create something more meaningful and helpful. 

The neuron_density and seed_neurons parameters can help us scale up the recurrent neural network while we stay faithful to the topology of the empirical neural network (here, the macaque monkey).

```
import numpy as np
neuron_density=np.zeros(29, dtype=int)
neuron_density[:] = 10

net_orig, net_scaled, region_neuron_ids = importnet.from_conn_mat(
    path_to_connectome_folder = path_to_connectome_folder, 
    data_name=data_name,
    neuron_density=neuron_density, 
    seed_neurons=None, 
    intrinsic_conn=False, 
    target_sparsity=0.1
    )
```
Now the neuron_density parameter is a numpy array and each entry is containing the number 10. This means that each region neuron_density[i] consists of 10 neurons. Thus, now the resulting recurrent neural network net_scaled contains 290 neurons (29 regions of the original connectome x 10 neurons per region as we indicated). These neurons are connected based on the topology of the the actual empirical neural network. Therefore, net_scaled is a bioinstantiated recurrent neural network, but scaled up to 290 neurons. 

If we want to assume that regions contain another number of neurons, we just simply construct neuron_density accordingly (e.g., with 20, 34, 1093 neurons, that is, arbitrary positive integers).

Note that not all regions need to contain the same number of neurons. For instance, we can assume that region 5 contains 40 neurons and the rest of the regions 10 neurons:

```
neuron_density=np.zeros(29, dtype=int)
neuron_density[:] = 10
neuron_density[4] = 40

net_orig, net_scaled, region_neuron_ids = importnet.from_conn_mat(
    path_to_connectome_folder = path_to_connectome_folder, 
    data_name=data_name,
    neuron_density=neuron_density, 
    seed_neurons=None, 
    intrinsic_conn=False, 
    target_sparsity=0.1
    )
```
This means each neuron_density[i] can contain an arbitrary positive integer. The total number of neurons, and thus, shape of net_scaled, for this example, has shape (320, 320).

Note that the parameter target_sparsity is a float (0 1] and controls for each source neuron the percentage of all possible neuron-targets to form connections with. Note that at least 1 neuron will function as target in case that the resulting percentage result in less than 1 neuron. This parameter can be used to make the sparsity of network_scaled vary around the density dictated by the actual biological connectomes. Default=0.2. Note that this parameter is meaningful only if at least one region has more than 1 neuron, that is, for some i, neuron_density[i]>1.

In such cases, the sparsity of network_scaled is affected by the parameter target_sparsity. For instance in the example above, the density of network_scaled (=0.066) corresponds to 6770 connections.

If target_sparsity=0.8 as in the example below:

```
net_orig, net_scaled, region_neuron_ids = importnet.from_conn_mat(
    path_to_connectome_folder = path_to_connectome_folder, 
    data_name=data_name,
    neuron_density=neuron_density, 
    seed_neurons=None, 
    intrinsic_conn=False, 
    target_sparsity=0.8
    )
```
then the density of network_scaled becomes higher(=0.529) corresponding to 54160 connections for net_scaled. Note that density of a network is the percentage of existing connections over the number of possible connections (given the shape of the array representing it, that is, network_scaled in this example), including the intrinsic, within-region connections and the self-to-self connections. 

# Intrinsic and self connections
If we want to build a neural network with intrinsic, within-region connections and self-to-self connections, then the intrinsic_conn and keep_diag parameters should be used:
```
net_orig, net_scaled, region_neuron_ids = importnet.from_conn_mat(
    path_to_connectome_folder = path_to_connectome_folder, 
    data_name=data_name,
    neuron_density=neuron_density, 
    seed_neurons=None, 
    intrinsic_conn=True, 
    target_sparsity=0.8,
    keep_diag=True,
    )   
```
Since the parameters intrinsic_conn and keep_diag are True, net_scaled exhibits both self-to-self connections and intrinsic, within-region connections. The network now has a density=0.572 and a total number of connections=58560.

Note that when the intrinsic_conn=True option is used, an additional parameter target_sparsity_intrinsic can be used to specify the sparsity of targets of the intrinsic connections (in the same way that target_sparsity works for the extrinsic connections). Default target_sparsity_intrinsic = 1. 

Let's see an example:
```
net_orig, net_scaled, region_neuron_ids = importnet.from_conn_mat(
   path_to_connectome_folder = path_to_connectome_folder, 
   data_name=data_name,
   neuron_density=neuron_density, 
   seed_neurons=None, 
   intrinsic_conn=True, 
   target_sparsity=0.8,
   target_sparsity_intrinsic = .5,
   keep_diag=True
   )
```
The network now has a density=0.550 and a total number of connections=56360.

When intrinsic_conn=True, the the parameter intrinsic_wei is relevant: intrinsic_wei is a float (0 1] denoting the percentage of the weight that will be assigned to the intrinsic weights. Default value intrinsic_conn=0.8, thus, 0.8*sum(extrinsic weight), where sum(extrinsic weight) is the sum of weights of connections from region A to all other regions, except from A. Note that the default value was used in all of the examples above. 

We can change the weight of intrinsic conenctions, including self-to-self connections, with the parameter intrinsic_wei, e.g., intrinsic_wei=0.5 in the example below:
```
net_orig, net_scaled, region_neuron_ids = importnet.from_conn_mat(
   path_to_connectome_folder = path_to_connectome_folder, 
   data_name=data_name,
   neuron_density=neuron_density, 
   seed_neurons=None, 
   intrinsic_conn=True, 
   target_sparsity=0.8,
   target_sparsity_intrinsic=0.5,
   keep_diag=True
   )
```
# Extrapolating conenction weights from empirical neural networks
In all of the above examples the parameter rand_partition is False (default values) and, thus, the neuron-to-neuron weights for each neuron m and n belonging to regions i and j respectively are computed as follows:
network_scaled[m,n] = network_original[i,j] / (nr_source_neurons * nr_target_neurons) with m, n all neurons belonging to region i, j respectively. 

This introduces less diversity and may impact the amount of diverse transformations that are applied to the input to the network (since many connections from one source neuron to many target neurons has the exact same weight). Therefore, in orderr to introduce more diversity but also construct net_scaled based on the empirical values of the neural network, we can specify rand_partition=True. This has as a result to compute heterogeneous strengths of for the neuron-to-neuron connections as follows: empirical wieight network_original[i,j] will be partitioned in k parts that sum to the original connection weight network_original[i,j], where k = nr_source_neurons * nr_target_neurons, with nr_source_neurons and nr_target_neurons are the nr of neurons in the source i and target areas j as part of network_scaled[i,j]. These k values will be assigned as conenction weights for the k connections between neurons inhabiting regions i and j. Let's see an example:
```
net_orig, net_scaled, region_neuron_ids = importnet.from_conn_mat(
   path_to_connectome_folder = path_to_connectome_folder, 
   data_name=data_name,
   neuron_density=neuron_density, 
   seed_neurons=None, 
   intrinsic_conn=True, 
   target_sparsity=0.8,
   target_sparsity_intrinsic=0.5,
   keep_diag=True,
   rand_partition=True
   )    
```
If seed_neurons is not None, but a positive integer, then the array neuron_density will be scaled such as neuron_density[i]/sum(neuron_density). Subsequently each entry neuron_density, will be multiplied by the seed_neurons integer. Thus, now each region neuron_density[i] contains neuron_density[i]/sum(neuron_density). This is derived from the following relation: neuron_density[i]/sum(neuron_density) * seed_neurons (actually, the ceil of this number, since we cannot have non-integer number of neurons in a region).

For instance, assuming 10 neurons per region and seed_neurons=100:

```
neuron_density=np.zeros(29, dtype=int)
neuron_density[:] = 10

net_orig, net_scaled, region_neuron_ids = importnet.from_conn_mat(
    path_to_connectome_folder = path_to_connectome_folder, 
    data_name=data_name,
    neuron_density=neuron_density, 
    seed_neurons=100, 
    intrinsic_conn=True, 
    target_sparsity=0.1
    )
```
The recurrent artifical neural network is now a network (net_scaled) containing in total 116 neurons.

Note that if neuron_density=None, then internally this will be set to: neuron_density[i]=1. Since neuron_density is scaled to the sum(neuron_density), instantiating the artifical recurrent neural network with neuron_density=None will result in the exact same number of neurons (=116) as the example above:  

```
net_orig, net_scaled, region_neuron_ids = importnet.from_conn_mat(
    path_to_connectome_folder = path_to_connectome_folder, 
    data_name=data_name,
    neuron_density=None, 
    seed_neurons=100, 
    intrinsic_conn=True, 
    target_sparsity=0.1
    )
```
The same syntax and parameters are used for instantiating the artifical recurrent neural network based on the topology of other empirical biological neural networks, such as the mouse:

```
data_name = "Mouse_Ypma_Oh"# the mouse neural network 

neuron_density=np.zeros(56, dtype=int)# this mouse network has 56 regions (see bio2art_from_conn_mat function documentation)
neuron_density[:] = 10

net_orig, net_scaled, region_neuron_ids = bio2art_import.bio2art_from_conn_mat(
    path_to_connectome_folder=path_to_connectome_folder, 
    data_name=data_name,
    neuron_density=neuron_density, 
    seed_neurons=None, 
    intrinsic_conn=True, 
    target_sparsity=0.1
    )
```
This instantiation results in a recurrent neural network net_scaled that contains 560 neurons (56 regions of the original connectome x 10 neurons per region as we indicated).

In all of the above examples net_orig is a ndarray that corresponds to the biological neural network that was used to construct the artificial neural network. region_neuron_ids is a list of lists. Each list in this list includes integers that are the indexes of the neurons contained within a region. For instance, region_neuron_ids[0] will return the indexes of the neurons in net_scaled that correspond to region 1 in the biological neural network net_orig. See section Utilities to see how the names of each egion can be imported. 

Note that import_net contains also the function from_list. This function can be used to read a csv file that represents a connectome and output the connectome as ndarray. Not used in the current examples but useful to read neural network data in a csv form.

# Examples of use in the context of echo state networks

Two examples are included to showcase the use of the Bio2Art conversion in an actual context. Both example focus on a "memory" capacity of the network (both in the "examples" folder).

bio2art_reservoire_lag_memory.py

This example uses an echo state network with random topology and one with a bioinstantiated topology. Given an input sequence, the task is to predict the Nth lag of the sequence.

bio2art_reservoire_sequence_memory.py

This example uses an echo state network with random topology and one with a bioinstantiated topology. The task is a "working memory" task, that is, the network has to memorize a sequence of N numbers and after a "cue" to replay this sequence.

Note that the above examples use the following echo state network implementation:
https://github.com/fabridamicelli/echoes

However, any echo state network can be used, since the Bio2Art offers as output a recurrent neural network in the form of a Numpy array that can be pluged-in as the recurrent network in-between Win and Wout in echo state networks.

Note that the examples can be run with the requirements enlisted in requirements.txt.

# Citations

Apart from explicitly refering to this repository, certain empirical datasets are used as well. Thus, if you use a specific empirical connectome to instantiate a recurrent artifical neural network, please cite the following papers:

Fly:
A.-S. Chiang et al. Three-dimensional reconstruction of brain-wide wiring networks in Drosophila at single-cell resolution.Curr. Biol.21,1–11 (2011) https://doi.org/10.1016/j.cub.2010.11.056

Mouse Ypma Oh:
M. Rubinov, R. J. F. Ypma, et al. Wiring cost and topological participation of the mouse brain connectome.Proc. Natl. Acad. Sci. U.S.A. 112,10032–10037 (2015). https://doi.org/10.1073/pnas.1420315112

S.W. Oh et al. A mesoscale connectome of the mouse brain. Nature. 508,207–214 (2014). http://dx.doi.org/10.1038/nature13186

Mouse Gamanut:

R. Gămănuţ et al. The mouse cortical Connectome, characterized by an ultra-dense cortical graph, maintains specificity by distinct connectivity profiles. Neuron. 97, 698-715.e10 https://doi.org/10.1016/j.neuron.2017.12.037

Macaque monkey:

N. T. Markov et al. A weighted and directed interareal connectivity matrix for macaque cerebral cortex. Cereb. Cortex 24,17–36 (2014). https://doi.org/10.1093/cercor/bhs270 

Marmoset monkey:

P. Majka et al. Towards a comprehensive atlas of cortical connections in a primate brain: Mapping tracer injection studies of the common marmoset into a reference digital template. Journal of Comparative Neurology. 524,2161–2181 (2016). https://doi.org/10.1002/cne.24023

Human:
R. F. Betzel, D. S. Bassett, Specificity and robustness of long-distance connections in weighted, interareal connectomes. Proc. Natl. Acad. Sci. U.S.A. 115, E4880–E4889 (2018). https://doi.org/10.1073/pnas.1720186115 
