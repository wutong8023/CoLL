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
# Intro: text generation datasets, which are used in continual machine translation scenarios.
# Author: Tongtong Wu
# Time: Aug 4, 2021
"""
import torch
import numpy as np

from torch.utils.data import DataLoader, Dataset
from base_dataset import BaseDataset


class WMT15(BaseDataset):
    # todo: load fewrel dataset
    def __init__(self):
        super(WMT15, self).__init__()
        pass
