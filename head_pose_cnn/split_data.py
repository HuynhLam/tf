'''
Lam Huynh, Oulu 2017
Split training + test data and save to text file
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

with open("train.txt", "w") as train_txt, open("val.txt", "w") as val_txt:
    for path, subdirs, files in os.walk(r'/home/lhuynh/Desktop/data/dogs_vs_cats_redux_kernels_edition/train'):
        random.shuffle(files)
        train_data = files[:int(0.75 * len(files))]
        val_data = files[int(0.75 * len(files)):]

        label = lambda x: ' 0' if 'cat' in x else ' 1'

        for filename in train_data:
            f = os.path.join(path, filename)
            train_txt.write(str(f) + label(filename) + os.linesep)

        for filename in val_data:
            f = os.path.join(path, filename)
            val_txt.write(str(f) + label(filename) + os.linesep)
