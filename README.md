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

Bio2Art builds artifical recurrent neuronal networks by using the topology dictated by the aforementioned empirical neural networks and by extrapolating from the empirical data to scale up the artifical neural networks. 

For instance, if the empirical data correspond to a neural network involving 29 brain regions, then the resulting artificial recurrent neural network can be scaled up by assuming a certain number of neurons populating each region (see examples below and documentation of the bio2art_import.py function). Thus, the output can be an artificial recurrent neural network with an arbitrary number of neurons (e.g., >>29 brain regions), but, importantly, this networks obeys the topology of a desired biological neural network (or "connectome").   

The constructed artificial recurrent neural network is returned as a numpy array and, thus, can be used with virtually any type of artifical recurrent network, for instance, echo state networks.  

# Installation

Download or clone the repository. Open a terminal and change to the corresponding folder. Type:

```
pip install .
```
Note that the Bio2Art only uses numpy (tested with numpy==1.16.2). However, to use the examples (see below), further libraries are needed. Therefore, for executing the examples, create a virtual environment (e.g., with conda) with the requirements enlisted in the requirements.txt file in the "examples" folder.  

# Examples

Converting the macaque monkey neural network to a recurrent artifical neural network.

```
from bio2art import bio2art_import # import Bio2Art function bio2art_import 
from pathlib import Path

# path to where the "connectomes" folder is located (it is included with the current repository)
path_to_connectome_folder = Path("/Users/alexandrosgoulas/Data/work-stuff/python-code/development/Bio2Art/connectomes/")

file_conn = "C_Macaque_Normalized.npy"# the macaque monkey neuronal network (see bio2art_from_conn_mat for all names of available connectomes)

net_orig, net_rescaled, region_neuron_ids = bio2art_import.bio2art_from_conn_mat(
    path_to_connectome_folder, 
    file_conn, 
    neuron_density=None, 
    seed_neurons=None, 
    intrinsic_conn=True, 
    target_sparsity=0.1
    )
```
The neuron_density is the recurrent neural network based on the indicated empirical monkey neuronal network. However, since neuron_density=None and seed_neurons=None, C_Neurons is exactly the same with C, that is, the exact same empirical monkey neural network. Not very useful. Let's see how we can create something more meaningful and helpful. 

The neuron_density and seed_neurons parameters can help us scale up the recurrent neural network while we stay faithful to the topology of the empirical neural network (here, the macaque monkey).

```
import numpy as np
neuron_density=np.zeros(29,)
neuron_density[:] = 10

net_orig, net_rescaled, region_neuron_ids = bio2art_import.bio2art_from_conn_mat(
    path_to_connectome_folder, 
    file_conn, 
    neuron_density=neuron_density, 
    seed_neurons=None, 
    intrinsic_conn=True, 
    target_sparsity=0.1
    )
```
Now the neuron_density parameter is a numpy array and each entry is containing the number 10. This means that each region ND[i] consists of 10 neurons. Thus, now the resulting recurrent neural network net_rescaled contains 290 neurons (29 regions of the original connectome x 10 neurons per region as we indicated). These neurons are connected based on the topology of the the actual empirical neural network. Therefore, net_rescaled is a bioinstantiated recurrent neural network, but scaled up to 290 neurons. 

If we want to assume that regions contain another number of neurons, we just simply construct neuron_density accordingly (e.g., with 20, 34, 1093 neurons, that is, arbitrary positive integers).

Note that not all regions need to contain the same number of neurons. For instance, we can assume that region 5 contains 40 neurons and the rest of the regions 10 neurons:

```
ND=np.zeros(29,)
ND[:] = 10
ND[4] = 40

C, C_Neurons, Region_Neuron_Ids = b2a.bio2art_from_conn_mat(
    path_to_connectome_folder, 
    file_conn, 
    ND=ND, 
    SeedNeurons=None, 
    intrinsic_conn=True, 
    target_sparsity=0.1
    )
```
This means each ND[i] can contain an arbitrary positive integer.

If SeedNeurons is not None, but a positive integer, then the array ND will be scaled such as ND[i]/sum(ND). Subsequently each entry ND, will be multiplied by the SeedNeurons integer. Thus, now each region ND[i] contains ND[i]/sum(ND) * SeedNeurons neurons (actually, the ceil of this number).

For instance, assuming 10 neurons per region and SeedNeurons=100:

```
ND=np.zeros(29,)
ND[:] = 10

C, C_Neurons, Region_Neuron_Ids = b2a.bio2art_from_conn_mat(
    path_to_connectome_folder, 
    file_conn, 
    ND=ND, 
    SeedNeurons=100, 
    intrinsic_conn=True, 
    target_sparsity=0.1
    )
```
The recurrent artifical neural network is now a network containing in total 116 neurons.

Note that if ND=None, then ND[i]=1. Since ND is scaled to the sum(ND), instantiating the artifical recurrent neural network with ND=None will result in the exact same output as the example above:  

```
C, C_Neurons, Region_Neuron_Ids = b2a.bio2art_from_conn_mat(
    path_to_connectome_folder, 
    file_conn, 
    ND=None, 
    SeedNeurons=100, 
    intrinsic_conn=True, 
    target_sparsity=0.1
    )
```
The same syntax and parameters are used for isntantiating the artifical recurrent neural network based on the topology of other empirical biological neural network, such as the mouse:

```
file_conn = "C_Mouse_Ypma_Oh.npy"# the mouse neuronal network 

ND=np.zeros(56,)# this mouse network has 56 regions (see bio2art_from_conn_mat function documentation)
ND[:] = 10

C, C_Neurons, Region_Neuron_Ids = b2a.bio2art_from_conn_mat(
    path_to_connectome_folder, 
    file_conn, 
    ND=ND, 
    SeedNeurons=None, 
    intrinsic_conn=True, 
    target_sparsity=0.1
    )
```
This instantiation results in a recurrent neural network C_Neurons that contains 560 neurons (56 regions of the original connectome x 10 neurons per region as we indicated).

In all of the above examples C is a numpy array that corresponds to the biological neural network that was used to construct the artificial neural network. Region_Neuron_Ids is a list of lists. Each list in this list includes integers that are the indexes of the neurons contained within a region. For instance, Region_Neuron_Ids[0] will return the indexes of the neurons in C_Neurons that correspond to region 1 in the biological neural network C. 

Note that in all of the examples above, the self-to-self connections (the diagonal of the C_Neurons numpy array) are set to 0, that is, treated as non-existent. To generate an array with self-to-self connections, use the parameter keep_diag=True (default value, thus, implicitly used in the examples above is False):

```
file_conn = "C_Mouse_Ypma_Oh.npy"# the mouse neuronal network 

ND=np.zeros(56,)# this mouse network has 56 regions (see bio2art_from_conn_mat function documentation)
ND[:] = 10

C, C_Neurons, Region_Neuron_Ids = b2a.bio2art_from_conn_mat(
    path_to_connectome_folder, 
    file_conn, 
    ND=ND, 
    SeedNeurons=None, 
    intrinsic_conn=True, 
    target_sparsity=0.1,
    keep_diag=True
    )
```
Now C_Neurons will have self-to-self connections (non-zero diagonal entries). Note that this parameter is meaningful only when intrinsic_conn=True.

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
