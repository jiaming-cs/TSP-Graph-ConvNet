import os
import json
import argparse
import time

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight


# Remove warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore', SparseEfficiencyWarning)

from config import *
from utils.graph_utils import *
from utils.google_tsp_reader import GoogleTSPReader
from utils.plot_utils import *
from models.gcn_model import ResidualGatedGCNModel
from utils.model_utils import *

config_path = "configs/tsp20.json"

config = get_config(config_path)
print("Loaded {}:\n{}".format(config_path, config))
dtypeFloat = torch.FloatTensor
dtypeLong = torch.LongTensor
torch.manual_seed(1)

net = nn.DataParallel(ResidualGatedGCNModel(config, dtypeFloat, dtypeLong))
if torch.cuda.is_available():
    net.cuda()
print(net)

# Compute number of network parameters
nb_param = 0
for param in net.parameters():
    nb_param += np.prod(list(param.data.size()))
print('Number of parameters:', nb_param)

# Define optimizer
learning_rate = config.learning_rate
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
print(optimizer)

# num_nodes = config.num_nodes
# num_neighbors = config.num_neighbors
# batch_size = config.batch_size
# train_filepath = config.train_filepath
# dataset = iter(GoogleTSPReader(num_nodes, num_neighbors, batch_size, train_filepath))
# batch = next(dataset)

# # Convert batch to torch Variables
# x_edges = Variable(torch.LongTensor(batch.edges).type(dtypeLong), requires_grad=False)
# x_edges_values = Variable(torch.FloatTensor(batch.edges_values).type(dtypeFloat), requires_grad=False)
# x_nodes = Variable(torch.LongTensor(batch.nodes).type(dtypeLong), requires_grad=False)
# x_nodes_coord = Variable(torch.FloatTensor(batch.nodes_coord).type(dtypeFloat), requires_grad=False)
# y_edges = Variable(torch.LongTensor(batch.edges_target).type(dtypeLong), requires_grad=False)
# y_nodes = Variable(torch.LongTensor(batch.nodes_target).type(dtypeLong), requires_grad=False)

# # Compute class weights
# edge_labels = y_edges.cpu().numpy().flatten()
# edge_cw = compute_class_weight("balanced", classes=np.unique(edge_labels), y=edge_labels)
# print("Class weights: {}".format(edge_cw))
    
# # Forward pass
# y_preds, loss = net.forward(x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges, edge_cw)
# loss = loss.mean()
# print("Output size: {}".format(y_preds.size()))
# print("Loss value:", loss)


# optimizer.zero_grad()
# loss.backward()

# # Optimizer step
# optimizer.step()

# # Compute error metrics 
# err_edges, err_tour, err_tsp, tour_err_idx, tsp_err_idx = edge_error(y_preds, y_edges, x_edges)
# print("Edge error: {:.3f}\nTour error: {:.3f}\nTSP error: {:.3f}".format(err_edges, err_tour, err_tsp))

# # Compute mean predicted and groundtruth tour length
# pred_tour_len = mean_tour_len_edges(x_edges_values, y_preds)
# gt_tour_len = np.mean(batch.tour_len) 
# print("Predicted tour length: {:.3f}\nGroundtruth tour length: {:.3f}".format(pred_tour_len, gt_tour_len))

log_dir = './data/tsp20/'
    # Load checkpoint
    # log_dir = f"./logs/{config.expt_name}/"
if torch.cuda.is_available():
    checkpoint = torch.load(log_dir+"best_val_checkpoint.tar")
else:
    checkpoint = torch.load(log_dir+"best_val_checkpoint.tar", map_location='cpu')
# Load network state
net.load_state_dict(checkpoint['model_state_dict'])
# Load optimizer state
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# Load other training parameters
epoch = checkpoint['epoch']
train_loss = checkpoint['train_loss']
val_loss = checkpoint['val_loss']
for param_group in optimizer.param_groups:
    learning_rate = param_group['lr']
print(f"Loaded checkpoint from epoch {epoch}")  


    # Set evaluation mode
net.eval()

batch_size = 10
num_nodes = config.num_nodes
num_neighbors = config.num_neighbors
beam_size = config.beam_size
test_filepath = config.test_filepath
dataset = iter(GoogleTSPReader(num_nodes, num_neighbors, batch_size, test_filepath))
batch = next(dataset)

with torch.no_grad():
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
    y_preds, loss = net.forward(x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges, edge_cw)
    loss = loss.mean()

    print(y_preds.shape)
    
    # Get batch beamsearch tour prediction
    bs_nodes = beamsearch_tour_nodes_shortest(
        y_preds, x_edges_values, beam_size, batch_size, num_nodes, dtypeFloat, dtypeLong, probs_type='logits')
    #bs_nodes = beamsearch_tour_nodes(
    #    y_preds, beam_size, batch_size, num_nodes, dtypeFloat, dtypeLong, probs_type='logits')

    # Compute mean tour length
    pred_tour_len = mean_tour_len_nodes(x_edges_values, bs_nodes)
    gt_tour_len = np.mean(batch.tour_len)
    print("Predicted tour length: {:.3f} (mean)\nGroundtruth tour length: {:.3f} (mean)".format(pred_tour_len, gt_tour_len))

    # Sanity check
    for idx, nodes in enumerate(bs_nodes):
        if not is_valid_tour(nodes, num_nodes):
            print(idx, " Invalid tour: ", nodes)

    # Plot prediction visualizations
    plot_predictions_beamsearch(x_nodes_coord, x_edges, x_edges_values, y_edges, y_preds, bs_nodes, num_plots=batch_size)