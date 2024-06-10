import os
import shutil
import pickle
import torch
import networkx as nx
import matplotlib.pyplot as plt

#added by Behnaz
import numpy as np
from datasets.preprocess import get_subfolder_number,listdir_from_subdirs



def mkdir(path):
    if os.path.isdir(path):
        is_del = input('Delete ' + path + ' Y/N:')
        if is_del.strip().lower() == 'y':
            shutil.rmtree(path)
        else:
            exit()

    os.makedirs(path)


def load_graphs(graphs_path, graphs_indices=None):
    """
    Returns a list of graphs given graphs directory and graph indices (Optional)
    If graphs_indices are not provided all graphs will be loaded
    """

    graphs = []
    if graphs_indices is None:
        for name in os.listdir(graphs_path):
            if not name.endswith('.dat'):
                continue

            with open(graphs_path + name, 'rb') as f:
                graphs.append(pickle.load(f))
    else:
        for ind in graphs_indices:
            with open(graphs_path + 'graph' + str(ind) + '.dat', 'rb') as f:
                graphs.append(pickle.load(f))

    return graphs

def load_graphs_with_subdirs(graphs_path, len_graphs,subdir_size,graphs_indices=None):
  """
  Returns a list of graphs given graphs directory and graph indices (Optional)
  If graphs_indices are not provided all graphs will be loaded
  """
  graphs = []
  if len_graphs>subdir_size:
    for ind in graphs_indices:
      #with open(graphs_path + 'graph' + str(ind) + '.dat', 'rb') as f:
      with open(os.path.join(graphs_path, str(get_subfolder_number(int(ind),subdir_size))+"/") + 'graph' + str(ind) + '.dat', 'rb') as f:
        graphs.append(pickle.load(f))

  else:
    if graphs_indices is None:
      for name in os.listdir(graphs_path):
        if not name.endswith('.dat'):
          continue

        with open(graphs_path + name, 'rb') as f:
          graphs.append(pickle.load(f))
    else:
      for ind in graphs_indices:
        with open(graphs_path + 'graph' + str(ind) + '.dat', 'rb') as f:
          graphs.append(pickle.load(f))


  return graphs




def save_graphs(graphs_path, graphs):
    """
    Save networkx graphs to a directory with indexing starting from 0
    """

    
    print('in save_graph_list, printing generated graph...')
    subgraph = graphs[0]
    print('subgraph.number_of_nodes() = ', subgraph.number_of_nodes())
    #plot_graph(subgraph)
    print('printing nodes...')
    for nodex in subgraph.nodes(data=True): 
      print(nx.info(nodex))
    print('printing edges...')
    for edge in  subgraph.edges(data=True):
      print(nx.info(edge))
    
    for i in range(len(graphs)):
        with open(graphs_path + 'graph' + str(i) + '.dat', 'wb') as f:
            pickle.dump(graphs[i], f)


# Create Directories for outputs
def create_dirs(args):
    if args.clean_tensorboard and os.path.isdir(args.tensorboard_path):
        shutil.rmtree(args.tensorboard_path)

    if args.clean_temp and os.path.isdir(args.temp_path):
        shutil.rmtree(args.temp_path)

    if not os.path.isdir(args.model_save_path):
        os.makedirs(args.model_save_path)

    if not os.path.isdir(args.temp_path):
        os.makedirs(args.temp_path)

    if not os.path.isdir(args.tensorboard_path):
        os.makedirs(args.tensorboard_path)

    if not os.path.isdir(args.current_temp_path):
        os.makedirs(args.current_temp_path)

    #added by Behnaz
    if not os.path.isdir(args.graphs_save_path):
      os.makedirs(args.graphs_save_path)
    
    #added by Behnaz
    if not os.path.isdir(args.figure_save_path):
      os.makedirs(args.figure_save_path)


def save_model(epoch, args, model, optimizer=None, scheduler=None, **extra_args):
    if not os.path.isdir(args.current_model_save_path):
        os.makedirs(args.current_model_save_path)
    
    
    
    fname = args.current_model_save_path + \
        args.fname + '_' + str(epoch) + '.dat'

    '''
    fname = args.model_save_path + \
            args.fname + '_' + str(epoch) + '.dat'
    '''
    #print("in utils/save_model and fname : ",fname)

    checkpoint = {'saved_args': args, 'epoch': epoch}

    save_items = {'model': model}
    if optimizer:
        save_items['optimizer'] = optimizer
    if scheduler:
        save_items['scheduler'] = scheduler

    for name, d in save_items.items():
        save_dict = {}
        for key, value in d.items():
            save_dict[key] = value.state_dict()

        checkpoint[name] = save_dict

    if extra_args:
        for arg_name, arg in extra_args.items():
            checkpoint[arg_name] = arg

    torch.save(checkpoint, fname)


def load_model(path, device, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(path, map_location=device)

    for name, d in {'model': model, 'optimizer': optimizer, 'scheduler': scheduler}.items():
        if d is not None:
            for key, value in d.items():
                value.load_state_dict(checkpoint[name][key])

        if name == 'model':
            for _, value in d.items():
                value.to(device=device)


def get_model_attribute(attribute, path, device):
    fname = path
    checkpoint = torch.load(fname, map_location=device)

    return checkpoint[attribute]


def draw_graph_list(G_list, row, col, fname = 'figures/test', layout='spring', is_single=False,k=1,node_size=55,alpha=1,width=1.3):
    # # draw graph view
    # from pylab import rcParams
    # rcParams['figure.figsize'] = 12,3
    plt.switch_backend('agg')
    for i,G in enumerate(G_list):
        plt.subplot(row,col,i+1)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1,
                        wspace=0, hspace=0)
        # if i%2==0:
        #     plt.title('real nodes: '+str(G.number_of_nodes()), fontsize = 4)
        # else:
        #     plt.title('pred nodes: '+str(G.number_of_nodes()), fontsize = 4)

        # plt.title('num of nodes: '+str(G.number_of_nodes()), fontsize = 4)

        # parts = community.best_partition(G)
        # values = [parts.get(node) for node in G.nodes()]
        # colors = []
        # for i in range(len(values)):
        #     if values[i] == 0:
        #         colors.append('red')
        #     if values[i] == 1:
        #         colors.append('green')
        #     if values[i] == 2:
        #         colors.append('blue')
        #     if values[i] == 3:
        #         colors.append('yellow')
        #     if values[i] == 4:
        #         colors.append('orange')
        #     if values[i] == 5:
        #         colors.append('pink')
        #     if values[i] == 6:
        #         colors.append('black')
        plt.axis("off")
        if layout=='spring':
            pos = nx.spring_layout(G,k=k/np.sqrt(G.number_of_nodes()),iterations=100)
            # pos = nx.spring_layout(G)

        elif layout=='spectral':
            pos = nx.spectral_layout(G)
        # # nx.draw_networkx(G, with_labels=True, node_size=2, width=0.15, font_size = 1.5, node_color=colors,pos=pos)
        # nx.draw_networkx(G, with_labels=False, node_size=1.5, width=0.2, font_size = 1.5, linewidths=0.2, node_color = 'k',pos=pos,alpha=0.2)

        if is_single:
            # node_size default 60, edge_width default 1.5
            #nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='#336699', alpha=1, linewidths=0, font_size=0)
            nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='#336699', alpha=1, linewidths=0)
            nx.draw_networkx_edges(G, pos, alpha=alpha, width=width)
        else:
            #nx.draw_networkx_nodes(G, pos, node_size=1.5, node_color='#336699',alpha=1, linewidths=0.2, font_size = 1.5)
            nx.draw_networkx_nodes(G, pos, node_size=1.5, node_color='#336699',alpha=1, linewidths=0.2)
            nx.draw_networkx_edges(G, pos, alpha=0.3,width=0.2)

        # plt.axis('off')
        # plt.title('Complete Graph of Odd-degree Nodes')
        # plt.show()
    plt.tight_layout()
    plt.savefig(fname+'.png', dpi=600)
    plt.close()


def pick_connected_component_new(G):
    adj_list = G.adjacency_list()
    for id,adj in enumerate(adj_list):
        id_min = min(adj)
        if id<id_min and id>=1:
        # if id<id_min and id>=4:
            break
    node_list = list(range(id)) # only include node prior than node "id"
    G = G.subgraph(node_list)
    G = max(nx.connected_component_subgraphs(G), key=len)
    return G

def load_graph_list(fname,is_real=True):
    with open(fname, "rb") as f:
        graph_list = pickle.load(f)
    
    for i in range(len(graph_list)):
        #edges_with_selfloops = graph_list[i].selfloop_edges()
        edges_with_selfloops = list(nx.selfloop_edges(graph_list[i]))
        if len(edges_with_selfloops)>0:
            graph_list[i].remove_edges_from(edges_with_selfloops)
        if is_real:
            #graph_list[i] = max(nx.connected_component_subgraphs(graph_list[i]), key=len)
            graph_list[i] = max((graph_list[i].subgraph(c) for c in nx.connected_components(graph_list[i])), key=len)
            graph_list[i] = nx.convert_node_labels_to_integers(graph_list[i])
        else:
            graph_list[i] = pick_connected_component_new(graph_list[i])
    
    return graph_list