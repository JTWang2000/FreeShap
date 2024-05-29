import copy
import datetime
from functools import reduce
import os
import pickle
import random
import time

import numpy as np
from itertools import chain, combinations
from os.path import join
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

def softmax(x, beta = 1.0):

    """Compute softmax values for each sets of scores in x."""
    x = np.array(x)
    e_x = np.exp(beta * x)
    return e_x / e_x.sum()


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def create_exp_dir(exp_name, dataset, results_dir='DataSV_Exp'):
    ''' Set up entire experiment directory '''
    str_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M')
    exp_dir = 'Exp_{}_{}'.format(exp_name, str_time)
    exp_dir = join(dataset, exp_dir)
    os.makedirs(join(results_dir, exp_dir), exist_ok=True)
    return join(results_dir, exp_dir)

def save_result(dir, sys_log, data):
    with open(join(dir, 'sys_log.txt'), 'w') as file:
        [file.write(key + ' : ' + str(value) + '\n') for key, value in sys_log.items()]

    with open(join(dir, 'sys_log.pickle'), 'wb') as f:
        pickle.dump(sys_log, f)
    
    with open(join(dir, "data.pickle"), "wb") as f:
        pickle.dump(data, f)
    # torch.save(data, join(dir,"data.pt"))

def fake_minority_class(minority_class, data_indices, keep_ratio):
    """
    To simulate the scenario of having a minority class

    :param int minority_class: the specified class to create minority sample
    :param list data_indices: the indices for each class in the dataset
    :param float keep_ratio: the ratio of data points remained in creating minority
    """
    minority_indices = data_indices[minority_class]
    minority_num = len(minority_indices)
    keep_num = int(minority_num * keep_ratio)
    sampled_minority_indices = np.random.choice(minority_indices, keep_num, replace=False)
    data_indices[minority_class] = list(sampled_minority_indices)
    return data_indices

def one_class_lo(n_participants, n_class, n_data_points, lo_class, data_indices, lo_ratio=0.3, lo_participant_percent=1.0):
    """
    To reserve one class and assign the one class data exclusively to some participants

    :param int n_participants: number of participants
    :param int n_class: number of classes in the label
    :param int n_data_points: number of data points in a normal participant
    :param int lo_class: the specified class to leave out
    :param list data_indices: the indices for each class in the dataset
    :param float lo_ratio: the ratio of participants that holds the exclusive data
    :param float lo_participant_percent: the ratio of number of data points for exclusive party
    """
    n_lo_participants = max(1,int(n_participants * lo_ratio))
    n_normal_participants = n_participants - n_lo_participants
    lo_indices = data_indices[lo_class]
    normal_indices = reduce(lambda a,b: a+b, [data_indices[i] for i in range(n_class) if i != lo_class])
    random.shuffle(normal_indices)
    indices_list = []
    end_point = None
    n_lo_data_points = len(lo_indices) // n_lo_participants
    lo_n_data_points = max(n_lo_data_points, int(n_data_points*lo_participant_percent))
    for i in range(n_normal_participants):
        indices_list.append(normal_indices[(i*n_data_points):((i+1)*n_data_points)])
        end_point = (i+1)*n_data_points
    for i in range(n_lo_participants):
        lo_part = lo_indices[(i*n_lo_data_points):((i+1)*n_lo_data_points)]
        if lo_n_data_points > n_lo_data_points:
            normal_part = normal_indices[(end_point + i*(lo_n_data_points-n_lo_data_points)):(end_point + (i+1)*(lo_n_data_points-n_lo_data_points))]
        indices_list.append(normal_part + lo_part)
    return indices_list

def duplicate_model_optimizer(model, optimizer_fn, lr, regularization=True):
    new_model = copy.deepcopy(model).cuda()
    new_optimizer = optimizer_fn(new_model.parameters(), lr=lr, weight_decay=5e-3 if regularization else 0)
    return new_model, new_optimizer

def itr_merge(*itrs):
    for itr in itrs:
        for v in itr:
            yield v

def merge_dataloaders(train_loaders):
    """Merge multiple dataloaders to a single loader"""
    indices = np.concatenate([tmp.sampler.indices for tmp in train_loaders],axis=0)
    batch_size = train_loaders[0].batch_size
    train_dataset = train_loaders[0].dataset

    return DataLoader(train_dataset,
                        batch_size=batch_size, 
                        sampler=SubsetRandomSampler(indices))
