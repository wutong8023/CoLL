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
    def forward(self):
        pass
    
    @abstractmethod
    def init_parameter(self):
        pass
    
    @abstractmethod
    def save_model(self):
        pass
    
    @abstractmethod
    def load_model(self):
        pass
    
    @abstractmethod
    def deep_copy(self):
        pass
    
    @abstractmethod
    def get_gradient(self):
        pass
    
    
