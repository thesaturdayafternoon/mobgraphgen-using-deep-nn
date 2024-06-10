import os
import random
import shutil
from statistics import mean
import torch

import utils

from graphgen.train import predict_graphs as gen_graphs_dfscode_rnn
from baselines.graph_rnn.train import predict_graphs as gen_graphs_graph_rnn
from baselines.dgmg.train import predict_graphs as gen_graphs_dgmg
from utils import get_model_attribute, load_graphs, save_graphs

import metrics.stats
#added by Behnaz
import pickle
from datasets.preprocess import listdir_from_subdirs
from utils import load_graphs_with_subdirs

from build_real_mob_graph import save_graphs_for_any_dir

LINE_BREAK = '----------------------------------------------------------------------\n'


class ArgsEvaluate():
    def __init__(self):
        # Can manually select the device too
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model_path ='model_save/' +"DFScodeRNN_grid_2022-06-18 16:55:46/DFScodeRNN_grid_4220.dat"
        
        self.num_epochs = get_model_attribute(
            'epoch', self.model_path, self.device)

        # Whether to generate networkx format graphs for real datasets
        #self.generate_graphs = True
        self.generate_graphs = False
        self.count = 2000
        #self.batch_size = 32  # Must be a factor of count
        self.batch_size = 32  # Must be a factor of count

        self.metric_eval_batch_size = 256
        #self.metric_eval_batch_size = 192

        # Specific DFScodeRNN
        #the original in code is self.max_num_edges = 50
        self.max_num_edges = 50

        # Specific to GraphRNN
        self.min_num_node = 0
        #self.max_num_node = 40
        self.max_num_node = 40


        self.minimum_length=10
        self.dim_list=["geo","sem"]
        self.indexes=""
        

        self.dir_input = ''

        self.train_args = get_model_attribute(
            'saved_args', self.model_path, self.device)

        self.graphs_save_path = self.dir_input+'graphs/'
        self.current_graphs_save_path = self.graphs_save_path + self.train_args.fname + '_' + \
            self.train_args.time + '/' + str(self.num_epochs) + '/'

        self.generated_graphs_save_path = self.graphs_save_path + "generated/"

        self.current_temp_path="tmp/DFScodeRNN_grid_2022-06-18 16:55:46"





def patch_graph(graph):
    for u in graph.nodes():
        graph.nodes[u]['label'] = graph.nodes[u]['label'].split('-')[0]

    return graph


def generate_graphs(eval_args):
    """
    Generate graphs (networkx format) given a trained generative model
    and save them to a directory
    :param eval_args: ArgsEvaluate object
    """

    train_args = eval_args.train_args

    if train_args.note == 'GraphRNN':
        gen_graphs = gen_graphs_graph_rnn(eval_args)
    elif train_args.note == 'DFScodeRNN':
        gen_graphs = gen_graphs_dfscode_rnn(eval_args)
    elif train_args.note == 'DGMG':
        gen_graphs = gen_graphs_dgmg(eval_args)

    if os.path.isdir(eval_args.current_graphs_save_path):
        shutil.rmtree(eval_args.current_graphs_save_path)

    os.makedirs(eval_args.current_graphs_save_path)
    print("in evaluate/generate_graphs before save_graphs(eval_args.current_graphs_save_path = ",eval_args.current_graphs_save_path, "gen_graphs)")
    save_graphs(eval_args.current_graphs_save_path, gen_graphs)


    #added by Behnaz
    if not os.path.isdir(eval_args.generated_graphs_save_path):
      os.makedirs(eval_args.generated_graphs_save_path)


    #added by Behnaz
    with open(eval_args.generated_graphs_save_path+"generated_graphs_list.dat", "wb") as f:
      print("in evaluate/generate_graphs and saving graphs in ",eval_args.generated_graphs_save_path,"generated_graphs_list.dat")
      pickle.dump(gen_graphs, f)


def generate_mob_graphs(eval_args,graph_samples_path=None):
    train_args = eval_args.train_args

    if train_args.note == 'GraphRNN':
        gen_graphs = gen_graphs_graph_rnn(eval_args)
    elif train_args.note == 'DFScodeRNN':

        gen_graphs = gen_graphs_dfscode_rnn(eval_args)
    elif train_args.note == 'DGMG':
        gen_graphs = gen_graphs_dgmg(eval_args)
    
    if graph_samples_path is not None:
      #print("6")
      save_graphs_for_any_dir(graph_samples_path,"generated_sample_graphs",gen_graphs)


    return gen_graphs


def print_stats(
    node_count_avg_ref, node_count_avg_pred, edge_count_avg_ref,
    edge_count_avg_pred, degree_mmd, clustering_mmd, orbit_mmd,
    nspdk_mmd, node_label_mmd, edge_label_mmd, node_label_and_degree
):
    print('Node count avg: Test - {:.6f}, Generated - {:.6f}'.format(
        mean(node_count_avg_ref), mean(node_count_avg_pred)))
    print('Edge count avg: Test - {:.6f}, Generated - {:.6f}'.format(
        mean(edge_count_avg_ref), mean(edge_count_avg_pred)))

    print('MMD Degree - {:.6f}, MMD Clustering - {:.6f}, MMD Orbits - {:.6f}'.format(
        mean(degree_mmd), mean(clustering_mmd), mean(orbit_mmd)))
    print('MMD NSPDK - {:.6f}'.format(mean(nspdk_mmd)))
    print('MMD Node label - {:.6f}, MMD Edge label - {:.6f}'.format(
        mean(node_label_mmd), mean(edge_label_mmd)
    ))
    print('MMD Joint Node label and degree - {:.6f}'.format(
        mean(node_label_and_degree)
    ))
    print(LINE_BREAK)


if __name__ == "__main__":
    eval_args = ArgsEvaluate()
    train_args = eval_args.train_args
    print("in evaluate/main and train_args.current_dataset_path = ",train_args.current_dataset_path)
    print("in evaluate/main...")
    print('Evaluating {}, run at {}, epoch {}'.format(
        train_args.fname, train_args.time, eval_args.num_epochs))

    print("in evaluate/main and generate graphs...")
    if eval_args.generate_graphs:
        generate_graphs(eval_args)

    random.seed(123)

    graphs = []

    print("in evaluate/main and make a list of graphs_test_indices and graphs_train_indices from graphs in train_args.current_dataset_path = ",train_args.current_dataset_path)
    #for name in os.listdir(train_args.current_dataset_path):
    for name in listdir_from_subdirs(parentdir=train_args.current_dataset_path):
        if name.endswith('.dat'):
            graphs.append(len(graphs))


    random.shuffle(graphs)
    print("len(graphs) = ",len(graphs))
    
    graphs_test_indices = graphs[int(0.90 * len(graphs)):]
    graphs_train_indices = graphs[:int(0.90 * len(graphs))]


    graphs_pred_indices = []
    if not eval_args.generate_graphs:
        for name in os.listdir(eval_args.current_graphs_save_path):
            if name.endswith('.dat'):
                graphs_pred_indices.append(len(graphs_pred_indices))
    else:
        graphs_pred_indices = [i for i in range(eval_args.count)]

    print('Evaluating {}, run at {}, epoch {}'.format(
        train_args.fname, train_args.time, eval_args.num_epochs))

    print('Graphs generated - {}'.format(len(graphs_pred_indices)))

    print("in evaluate/main and metrics.stats.novelity with train_args.current_dataset_path (train graphs dir) =",train_args.current_dataset_path," and eval_args.current_graphs_save_path(pred graphs dir) = ",eval_args.current_graphs_save_path)
    

    
    metrics.stats.novelity(
        train_args.current_dataset_path, graphs_train_indices, eval_args.current_graphs_save_path,
        graphs_pred_indices, train_args.temp_path, len(graphs),train_args.subdir_size,timeout=60)

    metrics.stats.uniqueness(eval_args.current_graphs_save_path,graphs_pred_indices, train_args.temp_path, timeout=120)




    # exit()

    node_count_avg_ref, node_count_avg_pred = [], []
    edge_count_avg_ref, edge_count_avg_pred = [], []

    degree_mmd, clustering_mmd, orbit_mmd, nspdk_mmd = [], [], [], []
    node_label_mmd, edge_label_mmd, node_label_and_degree = [], [], []

    print(len(graphs_test_indices))
    for i in range(0, len(graphs_pred_indices), eval_args.metric_eval_batch_size):
        #added by Behnaz
        print("in for loop of evaluate/main and i=", i)


        batch_size = min(eval_args.metric_eval_batch_size,len(graphs_pred_indices) - i)



        #added by behnaz
        print("in evaluate/main and batch_size = ",str(batch_size),"and len(graphs_test_indices) = ", len(graphs_test_indices), "and len(graphs_pred_indices) = ", len(graphs_pred_indices))
        graphs_ref_indices = random.sample(graphs_test_indices, batch_size)

        print("in evaluate/main and before graphs_ref = load_graphs(train_args.current_dataset_path=",train_args.current_dataset_path, "graphs_ref_indices=",graphs_ref_indices,")")


        graphs_ref = load_graphs_with_subdirs(train_args.current_dataset_path, len(graphs),train_args.subdir_size,graphs_ref_indices)

        graphs_ref = [patch_graph(g) for g in graphs_ref]

        graphs_pred = load_graphs(
            eval_args.current_graphs_save_path, graphs_pred_indices[i: i + batch_size])

        graphs_pred = [patch_graph(g) for g in graphs_pred]

        node_count_avg_ref.append(mean([len(G.nodes()) for G in graphs_ref]))
        node_count_avg_pred.append(mean([len(G.nodes()) for G in graphs_pred]))

        edge_count_avg_ref.append(mean([len(G.edges()) for G in graphs_ref]))
        edge_count_avg_pred.append(mean([len(G.edges()) for G in graphs_pred]))

        degree_mmd.append(metrics.stats.degree_stats(graphs_ref, graphs_pred))
        clustering_mmd.append(
            metrics.stats.clustering_stats(graphs_ref, graphs_pred))
        orbit_mmd.append(metrics.stats.orbit_stats_all(
            graphs_ref, graphs_pred))
        
        #added by behnaz
        print("In evaluate/main and len(graphs_ref) = ",len(graphs_ref)," len(graphs_pred) =",len(graphs_pred))
        nspdk_mmd.append(metrics.stats.nspdk_stats(graphs_ref, graphs_pred))

        node_label_mmd.append(
            metrics.stats.node_label_stats(graphs_ref, graphs_pred))
        edge_label_mmd.append(
            metrics.stats.edge_label_stats(graphs_ref, graphs_pred))
        node_label_and_degree.append(
            metrics.stats.node_label_and_degree_joint_stats(graphs_ref, graphs_pred))

        print('Running average of metrics:\n')

        print_stats(
            node_count_avg_ref, node_count_avg_pred, edge_count_avg_ref, edge_count_avg_pred,
            degree_mmd, clustering_mmd, orbit_mmd, nspdk_mmd, node_label_mmd,
            edge_label_mmd, node_label_and_degree
        )

    print('Evaluating {}, run at {}, epoch {}'.format(
        train_args.fname, train_args.time, eval_args.num_epochs))

    print_stats(
        node_count_avg_ref, node_count_avg_pred, edge_count_avg_ref, edge_count_avg_pred,
        degree_mmd, clustering_mmd, orbit_mmd, nspdk_mmd, node_label_mmd,
        edge_label_mmd, node_label_and_degree
    )
