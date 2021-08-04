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
import numpy as np

from torch.utils.data import DataLoader, Dataset
from base_dataset import BaseDataset

__all__ = ['Fewrel', 'SimpleQuestion', 'Tacred', 'Clinc150', 'Maven', 'Webred']


class Fewrel(BaseDataset):
    # todo: load fewrel dataset
    def __init__(self):
        super(Fewrel, self).__init__()
        pass


class SimpleQuestion(BaseDataset):
    # todo: load SimpleqQuestion dataset
    def __init__(self):
        super(SimpleQuestion, self).__init__()
        pass


class Tacred(BaseDataset):
    # todo: load tacred dataset
    def __init__(self):
        super(Tacred, self).__init__()
        pass


class Clinc150(BaseDataset):
    # todo: load clinc150 dataset
    def __init__(self):
        super(Clinc150, self).__init__()
        pass


class Maven(BaseDataset):
    # todo: load maven dataset
    def __init__(self):
        super(Maven, self).__init__()
        pass


class Webred(BaseDataset):
    # todo: load webred
    def __init__(self):
        super(Webred, self).__init__()
        pass


class Yelp(BaseDataset):
    # todo: load yelp
    def __init__(self):
        super(Yelp, self).__init__()
        pass
