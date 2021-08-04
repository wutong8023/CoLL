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
# Time: Aug 4, 2021
"""
from argparse import ArgumentParser
from base_method import BaseMethod


class Vanilla(BaseMethod):
    def __init__(self, backbone):
        super(Vanilla, self).__init__(backbone)
        pass
    
    def observe(self, x):
        pass
    
    def set_argument(self, parser) -> ArgumentParser:
        """
        set method-specific hyper-parameters
        """
        return parser