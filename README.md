# Bio2Art
Convert biological neuronal networks to artificial neuronal networks by creating 
recurrent neuronal networks based on the topology dictated by empirical connectomes.

![bio_and_art_connectomes](bio_and_art_connectomes.png)

# Description

The Bio2Art offers an easy to use function to convert biological neuronal networks to artificial recurrent neuronal networks. To this end, empirical neuronal networks of diverse species are used. Currently, the neuronal networks of the following species can be used:
1. Macaque monkey (Macaca mulatta)
2. Marmoset monkey (Callithrix jacchus)
3. Mouse (Mus musculus)
4. Human (Homo sapiens)
5. Fly (Drosophila melanogaster)

Note that the term 'connectome' refers to a biological neuronal network.

Bio2Art builds artifical recurrent neuronal networks by using the topology dictated by the aforementioned empirical and by extrapolating from the empirical data to scale up the artifical neuronal networks. For instance, if the empirical data correspond to a neuronal network involving 29 brain regions, then the resulting artificial recurrent neuronal network can be scaled up by assuming a certain number of neurons populating each region (see examples below and documentation of the bio2art_import.py function). Thus, the output can be an artificial recurrent neuronal network with an arbitrary number of neurons (e.g., >>29 brain regions), but, importantly this networks obeys the topology of a desired biological neuronal network.   

The constructed artificial recurrent neuronal network is returned as a numpy array and, thus, can be used with virtually any type of artifical recurrent network, for instance, echo state networks.  

# Installation

Download or clone the repository and unpack it. Open a terminal and change to the corresponding folder. Type:

```
pip install .
```
Note that the Bio2Art only uses Numpy (tested with numpy==1.16.2). However, to use the examples (see below), further libraries are needed. Therefore, for executing the examples, create a virtual environment (e.g., with conda) with the requirements enlisted in the requirements.txt file in the "examples" folder.  

# Examples

Converting the macaque monkey neuronal network to a recurrent artifical neuronal network.

```
# path to where the "conenctomes" folder is located (it is included with the current repository)
path_to_connectome_folder = Path("/Users/alexandrosgoulas/Data/work-stuff/python-code/development/Bio2Art/connectomes/")

file_conn = "C_Macaque_Normalized.npy"# the macaque monkey neuronal network (see bio2art_from_conn_mat for all names of available connectomes)

C, C_Neurons, Region_Neuron_Ids = bio2art_from_conn_mat(
    path_to_connectome_folder, 
    file_conn, 
    ND=None, 
    SeedNeurons=None, 
    intrinsic_conn=True, 
    target_sparsity=0.1
    )
```

# Citations 
