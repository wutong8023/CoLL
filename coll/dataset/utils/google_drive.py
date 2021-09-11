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

import requests
import zipfile
import os


def download_file(id: str, destination_dir: str, file_name: str = "target"):
    URL = "https://docs.google.com/uc?export=download"
    
    # get the path of output file
    if len(file_name.split(".")) == 1:
        file_name = file_name + ".zip"
    out_file = os.path.join(destination_dir, file_name)
    
    session = requests.Session()
    
    response = session.get(URL, params={'id': id}, stream=True)
    token = _get_confirm_token(response)
    
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    
    _save_response_content(response, out_file)
    
    if zipfile.is_zipfile(out_file):
        # check zip file
        zipped_file = zipfile.ZipFile(out_file)
        # unzip
        zipped_file.extractall(destination_dir)
        # remove zip file
        # os.remove(out_file)
    
    # todo: remove unused files and dirs
    # remove_path = os.path.join(destination_dir, "__MACOSX")
    # if os.path.exists(remove_path):
    #     # os.removedirs(remove_path)
    #     for f_path, dirs, fs in os.walk(remove_path):
    #         for f in fs:
    #             # print(os.path.join(f_path, f))
    #             os.remove(os.path.join(f_path, f))
    #         for dir in dirs:
    #             os.removedirs(os.path.join(f_path, dir))
    # # delete the target.zip
    # os.remove(out_file)


def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def _save_response_content(response, destination):
    CHUNK_SIZE = 32768
    
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
