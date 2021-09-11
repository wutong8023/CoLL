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

from argparse import ArgumentParser


class BaseMethod:
    def __init__(self, backbone):
        self.backbone = self.set_backbone(backbone)
        self.functions = ["before_stage", "observe", "after_stage"]
        self.capacity = {
            "task_boundary": False,
            "task_id_4_train": False,
            "task_id_4_test": False,
        }
        pass
    
    def set_backbone(self, backbone):
        backbone = backbone
        return backbone
    
    def before_stage(self):
        """
        stage-level preprocessing
        """
        pass
    
    def before_epoch(self):
        """
        epoch-level preprocessing
        """
        pass
    
    def observe(self, x):
        """
        batch-level processing
        """
        pass
    
    def after_epoch(self):
        """
        epoch-level postprocessing
        """
        pass
    
    def after_stage(self):
        """
        stage-level postprocessing
        """
        pass
    
    def set_argument(self, parser) -> ArgumentParser:
        """
        set method-specific hyper-parameters
        """
        return parser
