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
        self.train_stream = []
        self.test_stream = []
        self.current_stage = []
        self.stacked_test_data = []
        
        self.task_boundary = False
        self.task_id_4_train = False
        self.task_id_4_test = False
        pass
    
    def permutation(self):
        pass
    
    def learning_paradigm(self):
        pass
    

