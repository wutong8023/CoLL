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
# Intro: The training script for utilize experience replay
# Author: Tongtong Wu
# Time: Aug 3, 2021
"""

from coll.dataset.text_classification import Fewrel, Clinc150, Maven
from coll.stream.cre_controller import CreController
from coll.backbone.plm_classifier_backbone import PLMClassifier
from coll.train.classification_trainer import TCTrainer



# 1. define continual learning environment
# customize continual learning environment
data = FewRel()
paradigm = SemiSuper()
setting = TaskIL(split_by="clustering")
cl_env = Environment(data, paradigm, setting)

# or load predefined environment
cl_env = CreEnv()

# 2 define backbone model
backbone = PLMClassifier()

# 3 define continual learning strategy
memory = ReservoirMemory(size=500, extend=False)
cl_method = ER(memory)

# 4 train
Trainer.train(backbone, cl_env, cl_method)

# 5 evaluation
results = Evaluater.evaluate(backbone, cl_env, cl_method, acc_a)

print(results.summary())
