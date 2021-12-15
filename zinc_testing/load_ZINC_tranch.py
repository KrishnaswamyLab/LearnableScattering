from __future__ import print_function, division

import os, math, torch, pathlib

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, utils

import torch_geometric.data
from torch_geometric import utils
from torch_geometric import data

from pysmiles import read_smiles

from LEGS_module import Scatter


class ZINCDataset(Dataset):

    """ZINC Tranch data"""

    def __init__(self, file_name, transform=None, prop_stat_dict=None, include_ki=False):
        

        self.prop_list = ['qed', 'HeavyAtomMolWt', 'TPSA', 'RingCount']

        if include_ki:
            self.prop_list.append('Ki')
        
        self.tranch = np.load(file_name, allow_pickle=True).item()
        
        if prop_stat_dict != None:
            self.stats = np.load(prop_stat_dict, allow_pickle=True).item()
        else:
            self.stats = None

        self.transform = transform
        self.num_node_features = 1
        self.num_classes = len(self.prop_list)
        self.smi = list(self.tranch.keys())


    def __len__(self):
        
        return len(self.smi)

    
    def __getitem__(self, idx):         
        
        smi = self.smi[idx]

        props = np.zeros(self.num_classes)
        no_zscore = np.zeros(self.num_classes)

        if self.stats != None:
            #we want to zscore
            for i, entry in enumerate(self.prop_list):
                prop_value = self.tranch[smi][entry]
                z_scored = (prop_value - self.stats[entry]['mean']) / self.stats[entry]['std']
                props[i] = z_scored
        else:
            for i, entry in enumerate(self.prop_list):
                prop_value = self.tranch[smi][entry]
                props[i] = prop_value
                no_zscore[i] = prop_value

        mol = read_smiles(smi)
        data = from_networkx_custom(mol)

        data.no_zscore_props = no_zscore
        data.y = torch.Tensor([props])

        #place node features
        node_feats = []
    
        for i, entry in enumerate(data.element): 

            node_feat = np.zeros(self.num_node_features)
            node_feat[0] = 1.
            
            #one hot encoding of atoms       
            # if entry == 'C':
            #     node_feat[0] = 1.
            # elif entry == 'O':
            #     node_feat[1] = 1.
            # elif entry == 'N':
            #     node_feat[2] = 1.
            # elif entry == 'S':
            #     node_feat[3] = 1.
            
            # #pair encoding of atoms
            # if entry == 'C' or entry == 'O':
            #     node_feat[4] = 1.
            # if entry == 'C' or entry == 'N':
            #     node_feat[5] = 1.
            # if entry == 'C' or entry == 'S':
            #     node_feat[6] = 1.
            # if entry == 'O' or entry == 'N':
            #     node_feat[7] = 1.
            # if entry == 'O'  or entry == 'S':
            #     node_feat[8] = 1.
            # if entry == 'N' or entry == 'S':
            #     node_feat[9] = 1.

            node_feats.append(node_feat)

        data.x = torch.Tensor(node_feats)

        if self.transform: 
            return self.transform(data)
        else:
            return data


class Scattering(object):

    def __init__(self, scatter_model_name=None):

        model = Scatter(1, trainable_laziness=None)
        if scatter_model_name == None:
            raise ValueError("Please specify a pretrained scatter module. If you'd like to use an untrained model, specify\
            scatter_model_name='untrained'. Otherwise, use the .npy file of the model")
        elif scatter_model_name != 'untrained':
            model.load_state_dict(torch.load(scatter_model_name))
        model.eval()
        self.model = model
    
    def __call__(self, sample):

        props = sample.y
        to_return = self.model(sample)
        
        return to_return[0][0].detach(), sample.y[0]


def from_networkx_custom(G):

    r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.
    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
    """
    
    import networkx as nx

    G = nx.convert_node_labels_to_integers(G)
    G = G.to_directed() if not nx.is_directed(G) else G
    edge_index = torch.LongTensor(list(G.edges)).t().contiguous()

    data = {}

    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        for key, value in feat_dict.items():
            if(str(key) != "stereo"):
                data[str(key)] = [value] if i == 0 else data[str(key)] + [value]

    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        for key, value in feat_dict.items():
            data[str(key)] = [value] if i == 0 else data[str(key)] + [value]

    for key, item in data.items():
        try:
            data[key] = torch.tensor(item)
        except ValueError:
            pass

    data['edge_index'] = edge_index.view(2, -1)
    data = torch_geometric.data.Data.from_dict(data)
    data.num_nodes = G.number_of_nodes()

    return data