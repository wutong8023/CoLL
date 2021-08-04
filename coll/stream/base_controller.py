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


class BaseController:
    def __init__(self):
        self.datasets = []
        
        self.train_stream = []  # a list of training dataloader for each stage
        self.test_stream = []  # a list of testing dataloader respective to the current training stage
        
        self.current_stage = 0  # stage counter
        
        self.stacked_train_data = []  # for active-continual learning
        self.stacked_test_data = []  # for evaluation
        
        self.task_boundary = True  # if False: Online-learning
        self.task_id_4_train = True  # provide stage id during training
        self.task_id_4_test = True  # provide stage id during testing
        pass
    
    def permutation(self):
        """
        schedule the stage order
        """
        pass
    
    def learning_paradigm(self):
        """
        control the supervision of dataset to simulate:
        few-shot continual learning,
        semi-supervised continual learning,
        """
        pass
    
    def collate_fn(self):
        pass
    
    def prepare_dataloader(self):
        """
        split data into train / test;
        split dataset into multi-stage data stream;
        """
        pass

