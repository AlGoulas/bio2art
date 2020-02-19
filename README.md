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
import bio2art_import as b2a # import Bio2Art function bio2art_import 

# path to where the "connectomes" folder is located (it is included with the current repository)
path_to_connectome_folder = Path("/Users/alexandrosgoulas/Data/work-stuff/python-code/development/Bio2Art/connectomes/")

file_conn = "C_Macaque_Normalized.npy"# the macaque monkey neuronal network (see bio2art_from_conn_mat for all names of available connectomes)

C, C_Neurons, Region_Neuron_Ids = b2a.bio2art_from_conn_mat(
    path_to_connectome_folder, 
    file_conn, 
    ND=None, 
    SeedNeurons=None, 
    intrinsic_conn=True, 
    target_sparsity=0.1
    )
```
The C_Neurons is the reccurent neuronal network based on the indicated empirical monkey neuronal network. However, since ND=None and SeedNeurons=None, C_Neurons is exactly the same with C, that is, the exact same empirical monkey neuronal network. Not very useful. Let's see how the ND and SeedNeurons parameters can help us scale up the reccurent neuronal network while we stay faithful to the topology of the empirical neuronal network (here, the macaque monkey).

```
ND=np.zeros(29,)
ND[:] = 10

C, C_Neurons, Region_Neuron_Ids = b2a.bio2art_from_conn_mat(
    path_to_connectome_folder, 
    file_conn, 
    ND=None, 
    SeedNeurons=None, 
    intrinsic_conn=True, 
    target_sparsity=0.1
    )
```
Now the ND parameter is a numpy array and each entry is containing the numnber 10. This means that each region ND[i] consists of 10 neurons, Thus, now the resulting reccurent neuronal network C_Neurons contains 290 neurons (29 regions of the original connectome x 10 neurons per region as we indicated). These neurons are connected based on the topology of the the empirical actual neuronal network. Therefore, C_Neurons is a bioinstantiated eccurent neuronal network, but scaled up to 290 neurons. 

If we want to assume that regions contain another number of neurons, we jsut simply contruct ND accordingly (e.g., with 20, 34, 1093, etc neurons).

Note that not all region need to contain the same number of neurons. For isntance, we can assume that region 5 contains 40 neurons and the rest of the regions 10 neurons:

```
ND=np.zeros(29,)
ND[:] = 10
ND[4] = 40

C, C_Neurons, Region_Neuron_Ids = b2a.bio2art_from_conn_mat(
    path_to_connectome_folder, 
    file_conn, 
    ND=None, 
    SeedNeurons=None, 
    intrinsic_conn=True, 
    target_sparsity=0.1
    )
```
This means each ND[i] can contain an arbitrary positive integer.



# Citations 
