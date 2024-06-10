from datetime import datetime
import torch
from utils import get_model_attribute


class Args:
    """
    Program configuration
    """

    def __init__(self):
        # Can manually select the device too
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
            
        # Clean tensorboard
        self.clean_tensorboard = True
        # Clean temp folder
        self.clean_temp = True

        # Whether to use tensorboard for logging
        self.log_tensorboard = True

        # Algorithm Version - # Algorithm Version - GraphRNN  | DFScodeRNN (GraphGen) | DGMG (Deep GMG)
        self.note = 'DFScodeRNN'

        # Check datasets/process_dataset for datasets
        self.graph_type = 'grid'

        self.num_graphs = None  # Set it None to take complete dataset
        

        # Whether to produce networkx format graphs for real datasets
        self.produce_graphs = True
        # Whether to produce min dfscode and write to files
        self.produce_min_dfscodes = True
        self.produce_min_dfscode_tensors = True

        # if none, then auto calculate
        self.max_prev_node = None  # max previous node that looks back for GraphRNN

        # Specific to GraphRNN
        # Model parameters
        self.hidden_size_node_level_rnn = 128  # hidden size for node level RNN
        self.embedding_size_node_level_rnn = 64  # the size for node level RNN input
        self.embedding_size_node_output = 64  # the size of node output embedding
        self.hidden_size_edge_level_rnn = 16  # hidden size for edge level RNN
        self.embedding_size_edge_level_rnn = 8  # the size for edge level RNN input
        self.embedding_size_edge_output = 8  # the size of edge output embedding

        # Specific to DFScodeRNN
        # Model parameters
        self.hidden_size_dfscode_rnn = 256  # hidden size for dfscode RNN
        self.embedding_size_dfscode_rnn = 92  # input size for dfscode RNN
        # the size for vertex output embedding
        self.embedding_size_timestamp_output = 512
        self.embedding_size_vertex_output = 512  # the size for vertex output embedding
        self.embedding_size_edge_output = 512  # the size for edge output embedding
        self.dfscode_rnn_dropout = 0.2  # Dropout layer in between RNN layers
        #self.loss_type = 'BCE'
        self.loss_type = 'NLL'
        self.weights = False

        # Specific to DFScodeRNN
        self.rnn_type = 'LSTM'  # LSTM | GRU

        # Specific to GraphRNN | DFScodeRNN
        self.num_layers = 4  # Layers of rnn
        #added by Behnaz
        #self.num_layers = 6  # Layers of rnn

        self.batch_size = 32  # normal: 32, and the rest should be changed accordingly


        # Specific to DGMG
        # Model parameters
        self.feat_size = 128
        self.hops = 2
        self.dropout=0.2

        # training config
        #self.num_workers = 8  # num workers to load data, default 4
        self.num_workers = 8  # num workers to load data, default 4
        self.epochs = 30000

        self.lr = 0.003  # Learning rate
        # Learning rate decay factor at each milestone (no. of epochs)
        self.gamma = 0.3
        self.milestones = [100,200,400,800]  # List of milestones

        # Whether to do gradient clipping
        self.gradient_clipping = True

        # Output config
        self.dir_input = ''
        self.model_save_path = self.dir_input + 'model_save/'
        self.tensorboard_path = self.dir_input + 'tensorboard/'
        self.dataset_path = self.dir_input + 'datasets/'
        self.temp_path = self.dir_input + 'tmp/'

        self.figure_save_path = self.dir_input+'figures/'
        self.graphs_save_path = self.dir_input+'graphs/'
        self.real_graphs_save_path = self.graphs_save_path + "reals/"

        # Model save and validate parameters
        self.save_model = True
        #the line below is put in comment by Behnaz
        self.epochs_save = 20
        self.epochs_validate = 1
        #added by Behnaz
        self.epochs_print_loss = 20

        # Time at which code is run
        self.time = '{0:%Y-%m-%d %H:%M:%S}'.format(datetime.now())

        # Filenames to save intermediate and final outputs
        self.fname = self.note + '_' + self.graph_type

        
        self.current_model_save_path = self.model_save_path + \
            self.fname + '_' + self.time + '/'

        self.current_dataset_path = "datasets/grid/graphs"
        self.current_processed_dataset_path = None
        self.current_min_dfscode_path = None
        self.current_temp_path = self.temp_path + self.fname + '_' + self.time + '/'

        # Model load parameters
        self.load_model = True
        #self.load_model = False
        self.subdir_size=5000
        self.len_graphs=0
        self.graphs_created=True
        
        


        self.load_model_path ='model_save/' +"DFScodeRNN_grid_2022-06-18 16:55:46/DFScodeRNN_grid_4220.dat"

        #changed by Behnaz
        #self.load_device = torch.device('cuda:0')
        self.load_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            


        self.epochs_end = 30000

        #added by Behnaz
        self.city_names=["london2","london3"]
        self.second_sample_desired_size=2000
        self.chunk_size=10
        self.augmented_numbers={"london2":20,"london3":20}
        

    def update_args(self):
        if self.load_model:
            args = get_model_attribute(
                'saved_args', self.load_model_path, self.load_device)
            args.device = self.load_device
            args.load_model = True
            args.load_model_path = self.load_model_path
            args.epochs = self.epochs_end

            args.clean_tensorboard = False
            args.clean_temp = False

            args.produce_graphs = False
            args.produce_min_dfscodes = False
            args.produce_min_dfscode_tensors = False

            #added by Behnaz
            args.graphs_created=True


            args.lr = 0.003  # Learning rate
            args.gamma = 0.3
            args.dropout =  0.1
            args.dfscode_rnn_dropout = 0.05
            args.loss_type = 'NLL'


            return args

        return self
