import csv
import numpy as np

def list2sparse(targetList):
    indices = []
    vals = []
    for tI, target in enumerate(targetList):
        for seqI, val in enumerate(target):
            indices.append([tI, seqI])
            vals.append(val)
    shape = [len(targetList), np.asarray(indices).max(axis=0)[1]+1] #shape
    return (np.array(indices), np.array(vals), np.array(shape))

class DataProvider(object):
        """docstring for DataProvider"""
        def __init__(self, mode):
                super(DataProvider, self).__init__()

        def next(self, mode=None):
                lines = []
                with open('/data/ASR/corpus/librispeech/dataset/numpy/100h/train/character.csv', newline='') as csvfile:
                        reader = csv.reader(csvfile, delimiter=',')
                        for row in reader:
                                lines.append(row)

                inputs = [np.load(lines[1][2])]#, np.load(lines[2][2])]
                num_lists = [list(map(int, lines[1][3].split(' ')))]#, list(map(int, lines[2][3].split(' ')))]
                labels = list2sparse(num_lists)
                seq_lens = [int(lines[1][1])] #map(int, lines[2][1])]
                return inputs, labels, seq_lens

