import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight



# Remove warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore', SparseEfficiencyWarning)
import argparse

from config import *
from utils.graph_utils import *
from utils.google_tsp_reader import GoogleTSPReader
from utils.plot_utils import *
from models.gcn_model import ResidualGatedGCNModel
from utils.model_utils import *


class GraphConvNet():
    
    def __init__(self, problem):
        self.problem = problem        
        config_path = f"configs/{self.problem}.json"
        self.config = get_config(config_path)
        print("Loaded {}:\n{}".format(config_path, self.config))


    def load_model(self):

        torch.manual_seed(1)

        self.net = nn.DataParallel(ResidualGatedGCNModel(self.config, torch.FloatTensor, torch.LongTensor))
        if torch.cuda.is_available():
            self.net.cuda()
        print(self.net)

        # Compute number of network parameters
        nb_param = 0
        for param in self.net.parameters():
            nb_param += np.prod(list(param.data.size()))
        print('Number of parameters:', nb_param)

        # Define optimizer
        learning_rate = self.config.learning_rate
        optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        print(optimizer)


        log_dir = f'./data/{self.problem}/'
        
        if torch.cuda.is_available():
            checkpoint = torch.load(log_dir+"best_val_checkpoint.tar")
        else:
            checkpoint = torch.load(log_dir+"best_val_checkpoint.tar", map_location='cpu')
        # Load network state
        self.net.load_state_dict(checkpoint['model_state_dict'])
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Load other training parameters
        epoch = checkpoint['epoch']
        train_loss = checkpoint['train_loss']
        val_loss = checkpoint['val_loss']
        for param_group in optimizer.param_groups:
            learning_rate = param_group['lr']
        print(f"Loaded checkpoint from epoch {epoch}")  

        self.net.eval() # switch to evaluation model
        
    def get_edge_heatmap(self, kind):


        batch_size = 10
        num_nodes = self.config.num_nodes
        num_neighbors = self.config.num_neighbors
        beam_size = self.config.beam_size
        test_filepath = self.config.test_filepath
        dataset = iter(GoogleTSPReader(num_nodes, num_neighbors, batch_size, test_filepath))
        batch = next(dataset)

        with torch.no_grad():
            dtypeFloat = torch.FloatTensor
            dtypeLong = torch.LongTensor
            # Convert batch to torch Variables
            x_edges = Variable(torch.LongTensor(batch.edges).type(dtypeLong), requires_grad=False)
            x_edges_values = Variable(torch.FloatTensor(batch.edges_values).type(dtypeFloat), requires_grad=False)
            x_nodes = Variable(torch.LongTensor(batch.nodes).type(dtypeLong), requires_grad=False)
            x_nodes_coord = Variable(torch.FloatTensor(batch.nodes_coord).type(dtypeFloat), requires_grad=False)
            y_edges = Variable(torch.LongTensor(batch.edges_target).type(dtypeLong), requires_grad=False)
            y_nodes = Variable(torch.LongTensor(batch.nodes_target).type(dtypeLong), requires_grad=False)
            
            # Compute class weights
            edge_labels = y_edges.cpu().numpy().flatten()
            edge_cw = compute_class_weight("balanced", classes=np.unique(edge_labels), y=edge_labels)
            print("Class weights: {}".format(edge_cw))
            
            # Forward pass
            y_preds, loss = self.net.forward(x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges, edge_cw)
            loss = loss.mean()
            
            print(y_preds.shape)
            

            # Plot prediction visualizations
            plot_predictions(x_nodes_coord, x_edges, x_edges_values, y_edges, y_preds, kind, num_plots=batch_size)
            # plot_predictions_confused(x_nodes_coord, x_edges, x_edges_values, y_edges, y_preds, kind, num_plots=batch_size)
            
            
if __name__ == "__main__":
    
    
    '''
    Examples:
    python demo.py -m gd   //plot greedy search method result
    python demo.py -m pb   //plot branch purning + beam search method result
    python demo.py -m fr   //plot future reward method result
    '''
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--method", '-m', type=str, default='gd', help="serach method")
   
    args = parser.parse_args()

    if args.method == 'gd':
        kind = 1
    elif args.method == 'pb':
        kind = 3
    elif args.method == 'fr':
        kind = 2
    else:
        print("Invalid mehtod!")
        exit(0)
    gc = GraphConvNet("tsp50")
    gc.load_model()
    gc.get_edge_heatmap(kind)