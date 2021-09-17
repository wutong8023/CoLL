#!/usr/bin/env python
#
# Copyright the CoLL team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
# Intro: 
# Author: Tongtong Wu
# Time: Aug 3, 2021
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from argparse import ArgumentParser

__all__ = ['BaseDataset', 'collate_fn', 'print_data_structure', 'add_base_dataset_args']


def collate_fn(data):
    """
    collate data points as a batch.
    """
    xs = [item["x_source"] for item in data]
    ys = [item["y_source"] for item in data]
    
    x_tokens_id = torch.cat([item["x_tokens_id"] for item in data], dim=0)
    x_tokens_mask = torch.cat([item["x_tokens_mask"] for item in data], dim=0)
    
    y_tokens_id = torch.cat([item["y_tokens_id"] for item in data], dim=0)
    y_tokens_mask = torch.cat([item["y_tokens_mask"] for item in data], dim=0)
    
    y_id = torch.stack([item["y_id"] for item in data])
    
    return xs, ys, x_tokens_id, x_tokens_mask, y_tokens_id, y_tokens_mask, y_id


def print_data_structure():
    """
    instance example for all dataset
    """
    instance_example = {
        # x data
        "x_tokens_id": torch.Tensor,
        "x_source": [str],
        "x_tokens_mask": torch.Tensor,
        # y data
        "y_tokens_id": torch.Tensor,
        "y_source": [str],
        "y_tokens_mask": torch.Tensor,
        # y id
        "y_id": torch.Tensor,
    }
    print(instance_example)


# modularized arguments management
def add_base_dataset_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    parser.add_argument('--learning_paradigm', type=str, required=False,
                        help='the learning paradigm',
                        choices=["supervised", "active"], default=12)
    parser.add_argument('--data_filter_upper', type=int, required=False,
                        help='set the upper bound for instances per class',
                        default=10000)
    parser.add_argument('--data_filter_lower', type=int, required=False,
                        help='set the lower bound for instances per class',
                        default=10)
    parser.add_argument('--require_dev', type=bool, action='store_true',
                        help='whether need to split for development.')
    parser.add_argument('--train_ratio', type=float, required=False,
                        help='the ratio of data partition for training set.',
                        default=0.7)
    parser.add_argument('--test_ratio', type=float, required=False,
                        help='the ratio of data partition for testing set.',
                        default=0.3)
    args = parser.parse_args()
    if args.require_dev:
        parser.add_argument('--dev_ratio', type=float, required=False,
                            help='the ratio of data partition for dev set.',
                            default=0.1)


class BaseDataset(Dataset):
    def __init__(self, args):
        # file_path
        self.file_path = ""
        
        # link: google file id
        self.file_code = ""
        
        # dataset
        self.data = []
        self.labelled_data = []
        self.unlabelled_data = []
        
        # meta information
        self.target = None  # id for each instance, used for data partition.
        self.tokenizer = None  # related to the PLM
        self.label2id = {}  # used for classification task
        
        # whether need development dataset
        self.require_dev = False
        
        # learning paradigm
        self.paradigm = "supervised"
        if self.paradigm == "supervised":
            self.data = self.labelled_data
        else:
            self.data = self.unlabelled_data
        pass
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        instance = self.data[idx]
        return instance
    
    def download_data(self):
        """
        download data from the google drive.
        """
        pass
    
    def preprocess_data(self):
        """
        data preprocessing
        """
        pass
    
    def load_data(self):
        """
        load data from files.
        """
        pass
    
    def set_data(self, data, target=None, label2id=None):
        if label2id is not None:
            self.label2id = label2id
        self.data = data
        if target is None:
            self.targets = np.array([self.label2id[item["y_id"]] for item in self.data])
            


