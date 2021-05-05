from pickle import NONE
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
import time

from config import *
from utils.graph_utils import *
from utils.google_tsp_reader import GoogleTSPReader
from utils.plot_utils import *
from models.gcn_model import ResidualGatedGCNModel
from utils.model_utils import *
from TspSolvers import TspSolvers



class GraphConvNet():
    
    def __init__(self, problem):
        self.problem = problem        
        config_path = f"configs/{self.problem}.json"
        self.config = get_config(config_path)
        self.sovler = TspSolvers()
        #print("Loaded {}:\n{}".format(config_path, self.config))


    def load_model(self):

        torch.manual_seed(1)

        self.net = nn.DataParallel(ResidualGatedGCNModel(self.config, torch.FloatTensor, torch.LongTensor))
        if torch.cuda.is_available():
            self.net.cuda()
        #print(self.net)

        # Compute number of network parameters
        nb_param = 0
        for param in self.net.parameters():
            nb_param += np.prod(list(param.data.size()))
        #print('Number of parameters:', nb_param)

        # Define optimizer
        learning_rate = self.config.learning_rate
        optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        #print(optimizer)


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
        #print(f"Loaded checkpoint from epoch {epoch}")  

        self.net.eval() # switch to evaluation model
            
    def evaluation(self, batch_size, method, threshold, prob_weight, branch_limit):
        '''
        method: 
            'gd' = greedy
            'pb' = greedy + beam
            'fr' = future rewards
        '''
        if prob_weight == 1:
            prob_weight=None # if prob_weight is 1, we don't create new huruestic
        num_nodes = self.config.num_nodes
        num_neighbors = self.config.num_neighbors
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

            # Compute class weights
            edge_labels = y_edges.cpu().numpy().flatten()
            edge_cw = compute_class_weight("balanced", classes=np.unique(edge_labels), y=edge_labels)
            #print("Class weights: {}".format(edge_cw))
            
            # Forward pass
            start = time.time()
            y_preds, loss = self.net.forward(x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges, edge_cw)
            end = time.time()
            print("Network Overhead Time", end-start, "Batch Size", batch_size)
            loss = loss.mean()
            
            y = F.softmax(y_preds, dim=3)  # B x V x V x voc_edges
            y_probs = y[:,:,:,1]  # Prediction probabilities: B x V x V
            pred_total_len = 0
            gt_total_len = 0
            start = time.time()
            eva_size = batch_size
            for i in range(batch_size):
                W_val = x_edges_values[i].cpu().numpy()
                W_sol_probs = y_probs[i].cpu().numpy()
                if method == 'gd':
                    greedy_sol, _, ok = self.sovler.greedy_search(W_sol_probs, W_val, prob_weight)
                elif method == 'pb':
                    greedy_sol, _, ok = self.sovler.exclusive_search(W_sol_probs, W_val, threshold, branch_limit)
                elif method == 'fr':
                    greedy_sol, _, ok = self.sovler.future_reward(W_sol_probs)
                else:
                    print('Invalid method!')
                    break
                if not ok:
                    eva_size -= 1
                    continue
                pred_total_len += tour_nodes_to_tour_len(greedy_sol, W_val)
                gt_total_len += batch.tour_len[i]
                
            end = time.time()
            
            
            avg_pred = pred_total_len/eva_size
            avg_gt = gt_total_len/eva_size
            print("Average Predicted Tour Length:", avg_pred)
            print("Ground Truth Tour Length", avg_gt)
            opt_gap = max(0, (avg_pred-avg_gt)/avg_gt*100)
            
            print("Optimal Gap", opt_gap, "%")
            print("Execution Time", end-start, "s")

if __name__ == "__main__":
    
    '''
    Examples:
    python evaluation.py -s 20 -b 100 -m gd   //evluate greedy search method of 100 TSP20 problems
    python evaluation.py -s 50 -b 10 -m pb   //evluate branch pruning + beam search method of 100 TSP50 problems
    python evaluation.py -s 20 -b 100 -m fr   //evluate future rewards method of 100 TSP20 problems
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem_size", '-s', type=int, default=50, help="problem size")
    parser.add_argument("--method", '-m', type=str, default='gd', help="serach method")
    parser.add_argument("--threshold", '-t', type=float, default=0.3, help="branch pruning threshold")
    parser.add_argument("--batch_size", '-b', type=int, default=10, help="batch size")
    parser.add_argument("--branch_limit", '-l', type=int, default=4, help="maximum branch of brach pruning")
    parser.add_argument("--prob_weight", '-p', type=float, default=1, help="weight of probabiliy matrix")
    args = parser.parse_args()
    print("Problem Size:", "tsp"+str(args.problem_size))
    method = None
    if args.method == 'gd':
        method = "Gready Search"
    elif args.method == 'pb':
        method = "Branch Pruning + Beam Search"
    elif args.method == 'fr':
        method = "Future Reward"
    print("Method:", method)
    print("Threshold:", args.threshold)
    print("Batch Size:", args.batch_size)
    print("Threshold:", args.threshold)
    print("Branch Limit:", args.branch_limit)
    print("Probability Matrix Weight:", args.prob_weight)
    
    
    gc = GraphConvNet("tsp"+str(args.problem_size))
    gc.load_model()
    gc.evaluation(batch_size=args.batch_size, method=args.method, threshold=args.threshold, prob_weight=args.prob_weight, branch_limit=args.branch_limit)