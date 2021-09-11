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

from argparse import ArgumentParser

from coll.backbone.custom_head import ClassifierHead
from coll.backbone.base_backbone import BaseBackbone, add_base_model_args

__all__ = ['PLMClassifier', 'add_plm_classifier_args']


def add_plm_classifier_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    # adding the general arguments for base model.
    add_base_model_args(parser)
    
    # adding model-specific arguments
    parser.add_argument('--hidden_size', type=int, required=False,
                        help='the dimension of hidden embedding.',
                        default=200)
    parser.add_argument('--output_size', type=int, required=True,
                        help='the predefined maximum size of classes.', default='500')
    parser.add_argument('--expand', type=str, required=False,
                        help='when and how expand the architecture, the default is not changing',
                        choices=['fix', 'line', 'mix'],
                        default='fix')


class PLMClassifier(BaseBackbone):
    """
    A basic architecture for continual classification scenarios.
    """
    
    def __init__(self, args):
        super(PLMClassifier, self).__init__()
        self.PLM = args.PLM  # pytorch model
        self.head = ClassifierHead(input_size=args.hidden_size, hidden_size=args.hidden_size,
                                   out_size=args.output_size)
        self.Tokenizer = args.Tokenizer
        self.prob_layer = -1  # use the last layer for
    
    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, proto: torch.Tensor = None, task_id=None):
        encoding = self.PLM(x, attention_mask=x_mask, output_hidden_states=True)
        encoding = encoding.hidden_states[self.prob_layer]
        encoding = torch.mean(encoding, dim=1)
        output = self.head(encoding)
        return output
