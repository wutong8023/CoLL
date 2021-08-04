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
from base_backbone import BaseBackbone

__all__ = ['PLMClassifier']


class PLMClassifier(BaseBackbone):
    def __init__(self, plm, tokenizer, hidden_size=768, output_size=10):
        super(PLMClassifier, self).__init__()
        self.PLM = plm
        self.head = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Softmax(output_size)
        )
        self.Tokenizer = tokenizer
        pass
    
    def forward(self, x):
        pass
    
    def init_parameter(self):
        pass
    
    def save_model(self):
        pass
    
    def load_model(self):
        pass
    
    def deep_copy(self):
        pass
    
    def get_gradient(self):
        pass
