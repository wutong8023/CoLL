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
import matplotlib.pyplot as plt
import pandas
import os
import seaborn as sns


def visualize_grouped_bar(x_label, y_label, hue, title, data, file_path):
    sns.set_theme(style="whitegrid")
    
    # Draw a nested barplot by species and sex
    
    g = sns.catplot(
        data=data, kind="bar",
        x=x_label, y=y_label, hue=hue,
        ci="sd", palette="viridis", alpha=.6, height=6
    )
    
    sns.set(rc={"figure.dpi": 300, 'savefig.dpi': 300})
    g.despine(left=True)
    # plt.xlabel(x_label, fontsize=15)
    plt.ylabel(y_label, fontsize=15)
    g.legend.set_title(hue)
    
    # plt.title(title)
    if y_label == "ACCa":
        plt.ylim(0, 80)
    
    g.savefig(file_path)
    plt.clf()


data = pandas.read_csv("csv.csv")
data = pandas.DataFrame(data[data["metric"] == "ACCa"])
df = pandas.DataFrame(data[data["memory"] == 50])
x_label = "method"
y_label = "ACCa"
hu = "train"
file_path = "./acca50.png"
visualize_grouped_bar(x_label=x_label, y_label=y_label, hue=hu, data=df, file_path=file_path, title="")

data = pandas.read_csv("csv.csv")
data = pandas.DataFrame(data[data["metric"] == "ACCa"])
df = pandas.DataFrame(data[data["memory"] == 25])
x_label = "method"
y_label = "ACCa"
hu = "train"
file_path = "./acca25.png"
visualize_grouped_bar(x_label=x_label, y_label=y_label, hue=hu, data=df, file_path=file_path, title="")
