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
# Time: Jul 30, 2021
"""

import importlib
import math
import torch.nn as nn

# pretrained language model with 6 layers
PLM6 = {
    "distilbert": ["DistilBert", "distilbert-base-uncased"],
    "t5": ["T5", "t5-small"],
}

# pretrained language model with 12 layers
PLM12 = {
    "bert": ["Bert", "bert-base-uncased"],
    "roberta": ["Roberta", "roberta-base"],
    "albert": ["Albert", "albert-base-v2"],
    "xlnet": ["XLNet", "xlnet-base-cased"],
    "gpt2": ["GPT2", "gpt2"],
    "t5": ["T5", "t5-base"],
    "bart": ["Bart", "bart-base"],
}

# pretrained language model with 24 layers
PLM24 = {
    "bert": ["Bert", "bert-large-uncased"],
    "gpt2": ["GPT2", "gpt2-medium"],
    "gptneo": ["GPT2", "EleutherAI/gpt-neo-1.3B"],
    "xlnet": ["XLNet", "xlnet-large-cased"],
    "roberta": ["Roberta", "roberta-large"],
    "albert": ["Albert", "albert-large-v2"],
    "t5": ["T5", "t5-large"],
    "bart": ["Bart", "bart-large"],
}

# pretrained language model with 32 layers
PLM32 = {
    "gptneo": ["GPT2", "EleutherAI/gpt-neo-2.7B"],
}

# pretrained language model with 36 layers
PLM36 = {
    "gpt2": ["GPT2", "gpt2-large"],
}

# pretrained language model with 48 layers
PLM48 = {
    "gpt2": ["GPT2", "gpt2-xl"],
}


def get_plm(layer_num: int = 12):
    """
    return the pretrained language model
    """
    mod = importlib.import_module('coll.backbone')
    plm = getattr(mod, "PLM" + str(layer_num))
    plm = [key for key in plm.keys()]
    return plm


def xavier(m: nn.Module) -> None:
    """
    Applies Xavier initialization to linear modules.

    :param m: the module to be initialized

    Example::
        >>> net = nn.Sequential(nn.Linear(10, 10), nn.ReLU())
        >>> net.apply(xavier)
    """
    
    if m.__class__.__name__ == 'Linear':
        fan_in = m.weight.data.size(1)
        fan_out = m.weight.data.size(0)
        std = 1.0 * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        m.weight.data.uniform_(-a, a)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def import_from(module, name):
    """
    load module by name
    """
    module = __import__(module, fromlist=[name])
    return getattr(module, name)
