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
import torch.nn as nn

from abc import abstractmethod

__all__ = ['BaseBackbone']


class BaseBackbone(nn.Module):
    def __init__(self):
        super(BaseBackbone, self).__init__()
        self.prompt_encoder = None
        self.PLM = None
        self.Adapter = None
        self.head = None
        self.Tokenizer = None
        pass
    
    @abstractmethod
    def forward(self, x):
        pass
    
    def init_parameter(self, module):
        pass
    
    def save_model(self, dir_path):
        pass
    
    def load_model(self, model):
        pass
    
    def deep_copy(self):
        pass
    
    def get_gradient(self):
        pass
    
    def set_gradient(self, grad):
        pass
    
    def probing_layer(self, layer_id):
        pass