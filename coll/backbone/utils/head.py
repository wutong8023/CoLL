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
# Intro: the customized head for various nlp applications.
# Author: Tongtong Wu
# Time: Aug 4, 2021
"""
import torch
import torch.nn as nn

from coll.backbone import xavier

__all__ = ["ClassifierHead", "ProtoClassifierHead"]


class ClassifierHead(nn.Module):
    """
    general head for classification
    """
    
    def __init__(self, input_size=768, hidden_size=300, out_size=150, drop_out=0.8):
        super(ClassifierHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True),
            nn.Dropout(drop_out),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size, bias=True),
        )
        self.head.apply(xavier)
    
    def forward(self, x):
        x = self.head(x)
        return x


class ProtoClassifierHead(nn.Module):
    """
    prototype-based classification
    """
    
    def __init__(self):
        super(ProtoClassifierHead, self).__init__()
    
    def forward(self, x: torch.Tensor, prototype: torch.tensor):
        prototype = torch.transpose(prototype, 0, 1)
        output = torch.matmul(x, prototype)
        return output
