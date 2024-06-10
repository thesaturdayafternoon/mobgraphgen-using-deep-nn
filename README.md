## Publication

This repository is the official Python/PyTorch implementation of the evaluation part of the article written by Behnaz Bostanipour, which is submitted to the IEEE, 2024.

Some parts of the code has been adapted from [the official implementation of GraphGen](https://github.com/idea-iitd/graphgen).

## Overview

See the abstract of the article describing this work.

## Code Description

This repository contains two main folders: `mobgraphgen` and `Colab Notebooks`. The folder `mobgraphgen` contains the Python/PyTorch modules. The Google Colaboratory notebooks of `Colab Notebooks` use modules of `mobgraphgen` for performing different tasks. Below, we discuss each folder in detail.

`mobgraphgen`: as indicated in our paper, the present work is built 
upon [GraphGen](https://arxiv.org/pdf/2001.08184.pdf), a state-of-the art framework for developing deep generative models for graphs. Accordingly, most of the modules in this folder has been adapted from [the official implementation of GraphGen](https://github.com/idea-iitd/graphgen). 
We also added new modules, in particular:
- `build_real_mob_graph.py`, `map_info.py`, `maps_info.py`: these modules include code for building mobility graphs from 
real-world user mobility data (i.e., user check-ins stored in folders `input_london2` and `input_london3`), semantically 
augmenting the set of mobility graphs to increase its size, building uniformly random mobility graphs, 
print different information about mobility graphs, etc. Note that we use a dataset of Foursquare check-ins 
(collected through Twitter’s public stream) from the City of London. Due to the terms and conditions of Twitter 
and Foursqaure, the dataset is private. Accordingly, we do not publish the dataset and the mobility graphs built from it.  

- `new_tree.py`: by using this module and the information stored in `fs_tag_tree.txt` (a file containing the Foursquare 
tag hierarchy tree at the time of data collection), a semantic tag tree for Foursquare semantic tags is built in the constructor 
of `map_info.py` and then used for semantically augmenting the mobility graphs.
- `util_priv_eval/helper.py`: this contains the functions that are used for the utility and the privacy evaluation. 

`Colab Notebooks`: The notebooks in this folder are used for different purposes:

- `complete_privacy_evaluation.ipynb`: this is the notebook for the privacy evaluation (depicting the evaluation results). 
- `complete_utility_evaluation.ipynb`: this is the notebook for the utility evaluation (depicting the evaluation results). 
- `create_graph_datasets_and_train_the_model.ipynb`: this notebook can be used in different ways depending on 
some Booleans defined in `mobgraphgen/args.py`. In summary, by running this notebook, one can create the datasets required for 
training and evaluation of the model, train a new model, or resume the training of an existing model.
- `create_graphs_from_checkins.ipynb`: this notebook creates graphs from the user checkins and augments the original set of graphs. It also saves the created graphs, including the training set and the test set.
- `create_mindfscode_tensors.ipynb`: this notebook generates minimum dfscode tensors from minimum dfscodes and saves them.
- `create_mindfscodes.ipynb`: this notebook generates minimum dfscodes from graphs and saves them.
- `create_random_graphs.ipynb`: this notebook generates uniformly random graphs and saves them.
- `create_tar_files_for_mindfscode _tensors.ipynb`: this notebook generates and saves a tar file for each subdirectory in 
`mobgraphgen/datasets/grid/min_dfscide_tensors`. The tar files could be decompressed during the training into VM disk. Reading a file into the VM disk is so much faster than reading a subdirectory content from colab Drive.
- `generate_synthetic_graphs.ipynb`: this is the code for generating and saving a set of synthetic graphs using the trained model.

