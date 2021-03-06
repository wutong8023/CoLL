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

__all__ = ['ALL_DATASET', 'CONTINUAL_RE_DATASET', 'LAMOL_DATASET', 'CONTINUAL_TC_DATASET']

ALL_DATASET = {
    "fewrel": "relation extraction"
}

CONTINUAL_RE_DATASET = {
    "fewrel": "relation extraction"
}

LAMOL_DATASET = {
    "squad": "question answering",
    "wikisql": "text2sql"
}

CONTINUAL_TC_DATASET = {
    "yahoo": "sentiment classification",
    "yelp": "sentiment classification"
}
