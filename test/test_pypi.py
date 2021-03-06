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
# Intro: version-0.01 test
# Author: Tongtong Wu
# Time: Jul 30, 2021
"""
from firmiana import hello_world
import unittest

class TestStringMethods(unittest.TestCase):

    def test_pypi(self):
        hello_world()
        self.assertEqual(hello_world(), None)


if __name__ == '__main__':
    unittest.main()