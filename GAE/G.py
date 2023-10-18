#!/usr/bin/env python3
# -*- encoding: utf-8 -*-


'''
@Useway  :   存储所有的位置和全局变量
@File    :   G.py
@Time    :   2021/01/03 16:09:04
@Author  :   Chen Zhuang 
@Version :   1.0
@Contact :   whut_chenzhuang@163.com
@Time: 2021/01/03 16:09:04
'''

from pathlib import Path


DATA_PATH = Path('./data')

TRAIN_DATA_PATH = DATA_PATH.joinpath('train.npy')

VAL_DATA_PATH = DATA_PATH.joinpath('val.npy')

TEST_DATA_PATH = DATA_PATH.joinpath('test.npy')

OUT_DIR = Path('./weight')