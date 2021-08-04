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


__all__ = ['BaseDataset']

class BaseDataset(Dataset):
    def __init__(self):
        self.file_path = ""
        self.data = []
        self.target = None
        self.tokenizer = None
        pass
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        instance = self.data[idx]
        return instance
    
    def download_data(self):
        pass
    
    def load_data(self):
        pass
    
    def preprocess_data(self):
        pass
