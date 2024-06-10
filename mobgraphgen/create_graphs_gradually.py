import random
import time
import pickle
from torch.utils.data import DataLoader

from args import Args
from utils import create_dirs
from datasets.process_dataset import create_graphs
from datasets.preprocess import calc_max_prev_node, dfscodes_weights
from baselines.dgmg.data import DGMG_Dataset_from_file
from baselines.graph_rnn.data import Graph_Adj_Matrix_from_file
from graphgen.data import Graph_DFS_code_from_file
from model import create_model
from train import train
from evaluate import ArgsEvaluate
from evaluate import generate_graphs
#added by Bhenaz
import os
import gc
#from google.colab import drive
from utils import mkdir
from datasets.preprocess import (
    mapping, graphs_to_min_dfscodes,
    min_dfscodes_to_tensors, random_walk_with_restart_sampling
)
from datasets.preprocess import listdir_from_subdirs,save_graphs_dataset,create_sub_dirs
from build_real_mob_graph import build_graph_samples_for_cities
from datasets.process_dataset import produce_graphs_from_raw_format,produce_graphs_from_graphrnn_format,produce_random_walk_sampled_graphs
from build_real_mob_graph import create_random_uniform_graphs

def print_datadirs_len(args):
  if 'grid' in args.graph_type:
    base_path = os.path.join(args.dataset_path, 'grid/')
    # Node invariants - Options 'Degree' and 'CC'
    min_num_nodes, max_num_nodes = None, None
    min_num_edges, max_num_edges = None, None
  else:
    print('Dataset - {} is not valid'.format(args.graph_type))
    exit()

  args.current_dataset_path = os.path.join(base_path, 'graphs/')
  args.min_dfscode_path = os.path.join(base_path, 'min_dfscodes/')
  min_dfscode_tensor_path = os.path.join(base_path, 'min_dfscode_tensors/')
  #added by Behnaz
  graph_samples_path=os.path.join(base_path, 'graph_samples/')
    
  if args.note == 'GraphRNN' or args.note == 'DGMG':
    args.current_processed_dataset_path = args.current_dataset_path
  elif args.note == 'DFScodeRNN':
    args.current_processed_dataset_path = min_dfscode_tensor_path

  #mkdir(args.current_dataset_path)
  count = len([name for name in listdir_from_subdirs(parentdir=args.current_dataset_path) if name.endswith(".dat")])
  print('Number of Graphs', count)
  gc.collect()
  
  
  # Produce feature map
  with open(args.current_dataset_path + 'map.dict', 'rb') as f:
        feature_map = pickle.load(f)
  print("printing feature_map")
  print(feature_map)
  del feature_map
  gc.collect()
  

  min_dfscodes_filenames=listdir_from_subdirs(parentdir=args.min_dfscode_path)
  print("len(min_dfscodes_filenames) = ",len(min_dfscodes_filenames))
  del min_dfscodes_filenames
  gc.collect()

  min_dfscodes_tensors_filenames=listdir_from_subdirs(parentdir=min_dfscode_tensor_path)
  print("len(min_dfscodes_tensors_filenames) = ",len(min_dfscodes_tensors_filenames))
  del min_dfscodes_tensors_filenames
  gc.collect()



# Routine to create datasets
def create_min_dfscode_tensors(args):
    #added by Behnaz
    if 'grid' in args.graph_type:
        base_path = os.path.join(args.dataset_path, 'grid/')
        # Node invariants - Options 'Degree' and 'CC'
        
        min_num_nodes, max_num_nodes = None, None
        min_num_edges, max_num_edges = None, None

    else:
        print('Dataset - {} is not valid'.format(args.graph_type))
        exit()

    args.current_dataset_path = os.path.join(base_path, 'graphs/')
    args.min_dfscode_path = os.path.join(base_path, 'min_dfscodes/')
    #min_dfscode_tensor_path = os.path.join(base_path, 'min_dfscode_tensors/')
    min_dfscode_tensor_path = os.path.join(base_path, 'min_dfscode_tensors/')
    #added by Behnaz
    graph_samples_path=os.path.join(base_path, 'graph_samples/')
    
    if args.note == 'GraphRNN' or args.note == 'DGMG':
        args.current_processed_dataset_path = args.current_dataset_path
    elif args.note == 'DFScodeRNN':
        args.current_processed_dataset_path = min_dfscode_tensor_path


    count = len([name for name in listdir_from_subdirs(parentdir=args.current_dataset_path) if name.endswith(".dat")])
    #print('Graphs counted', count)
    with open(args.current_dataset_path + 'map.dict', 'rb') as f:
      feature_map = pickle.load(f)    

    mkdir(min_dfscode_tensor_path)
    create_sub_dirs(min_dfscode_tensor_path,len(next(os.walk(args.min_dfscode_path))[1]))
    start = time.time()
    min_dfscodes_to_tensors(args.min_dfscode_path,min_dfscode_tensor_path, feature_map)
    end = time.time()
    print('Time taken to make dfscode tensors= {:.3f}s'.format(end - start))
    min_dfscodes_tensors_filenames=listdir_from_subdirs(parentdir=min_dfscode_tensor_path)
    print("len(min_dfscodes_tensors_filenames) = ",len(min_dfscodes_tensors_filenames))
    del min_dfscodes_tensors_filenames
    gc.collect()
    

# Routine to create datasets
def create_min_dfscodes(args):
    if 'grid' in args.graph_type:
        base_path = os.path.join(args.dataset_path, 'grid/')
        # Node invariants - Options 'Degree' and 'CC'
        
        min_num_nodes, max_num_nodes = None, None
        min_num_edges, max_num_edges = None, None

    else:
        print('Dataset - {} is not valid'.format(args.graph_type))
        exit()

    args.current_dataset_path = os.path.join(base_path, 'graphs/')
    args.min_dfscode_path = os.path.join(base_path, 'min_dfscodes/')
    min_dfscode_tensor_path = os.path.join(base_path, 'min_dfscode_tensors/')
    #added by Behnaz
    graph_samples_path=os.path.join(base_path, 'graph_samples/')
    
    if args.note == 'GraphRNN' or args.note == 'DGMG':
        args.current_processed_dataset_path = args.current_dataset_path
    elif args.note == 'DFScodeRNN':
        args.current_processed_dataset_path = min_dfscode_tensor_path

    count = len([name for name in listdir_from_subdirs(parentdir=args.current_dataset_path) if name.endswith(".dat")])
    #print('Graphs counted', count)

    mkdir(args.min_dfscode_path)
    create_sub_dirs(args.min_dfscode_path,len(next(os.walk(args.current_dataset_path))[1]))
    start = time.time()
    graphs_to_min_dfscodes(args.current_dataset_path,args.min_dfscode_path, args.current_temp_path)
    end = time.time()
    print('Time taken to make dfscodes = {:.3f}s'.format(end - start))
    #added by Behnaz
    #print("args.min_dfscode_path = ",args.min_dfscode_path)
    #min_dfscodes_filenames=[name for name in os.listdir(args.min_dfscode_path) if name.endswith(".dat")]
    min_dfscodes_filenames=listdir_from_subdirs(parentdir=args.min_dfscode_path)
    print("len(min_dfscodes_filenames) = ",len(min_dfscodes_filenames))
    #print("len(min_dfscodes_filenames) = ",len(listdir_from_subdirs(parentdir=args.min_dfscode_path)))
    del min_dfscodes_filenames
    gc.collect()
    #drive.flush()



# Routine to create datasets
def create_graphs_only(args):

    if 'grid' in args.graph_type:
        base_path = os.path.join(args.dataset_path, 'grid/')
        # Node invariants - Options 'Degree' and 'CC'
        
        min_num_nodes, max_num_nodes = None, None
        min_num_edges, max_num_edges = None, None

    else:
        print('Dataset - {} is not valid'.format(args.graph_type))
        exit()

    args.current_dataset_path = os.path.join(base_path, 'graphs/')
    args.min_dfscode_path = os.path.join(base_path, 'min_dfscodes/')
    min_dfscode_tensor_path = os.path.join(base_path, 'min_dfscode_tensors/')
    #added by Behnaz
    graph_samples_path=os.path.join(base_path, 'graph_samples/')
    
    if args.note == 'GraphRNN' or args.note == 'DGMG':
        args.current_processed_dataset_path = args.current_dataset_path
    elif args.note == 'DFScodeRNN':
        args.current_processed_dataset_path = min_dfscode_tensor_path

    mkdir(args.current_dataset_path)

    #added by Behnaz     
    if args.graph_type in ['grid']:

      graphs=build_graph_samples_for_cities(args.city_names,args.second_sample_desired_size,graph_samples_path,args.chunk_size,args.augmented_numbers,change_indexes="")

      save_graphs_dataset(graphs,args,subdir_size=args.subdir_size) 
      with open(args.real_graphs_save_path+"real_graphs_list.dat", "wb") as f:
        pickle.dump(graphs, f) 
      del graphs
      gc.collect()
      create_random_uniform_graphs(args.city_names,args.chunk_size,args.second_sample_desired_size, graph_samples_path, change_indexes="")
      gc.collect()
      count = len([name for name in listdir_from_subdirs(parentdir=args.current_dataset_path) if name.endswith(".dat")])
      #print('Graphs produced', count)

      # Produce feature map
      feature_map = mapping(args.current_dataset_path,args.current_dataset_path + 'map.dict')
      #print(feature_map)
      gc.collect()
      #drive.flush()
      
    gc.collect()
    print(feature_map)
    
    #the original in code
    graphs = [i for i in range(count)]
    args.len_graphs=count
    gc.collect()
    return graphs





    

