import csv
import numpy as np

# Borrow from https://github.com/zzw922cn/Automatic_Speech_Recognition/
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
        if mode == 'train':
            self.csv_path = "/data/ASR/corpus/librispeech/dataset/numpy/100h/train/character.csv"
        if mode == 'dev':
            self.csv_path = "/data/ASR/corpus/librispeech/dataset/numpy/100h/dev_clean/character.csv"
        if mode == 'test':
            self.csv_path = "/data/ASR/corpus/librispeech/dataset/numpy/100h/test_clean/character.csv"

        # read csv file
        lines = []
        with open(self.csv_path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                lines.append(row)
        lines.pop(0) # get rid of header
        self.lines = lines
        self.data_size = len(lines)
        self.pos = 0

    def get_data_size(self):
        return self.data_size

    def next(self, batch_size=1):
        inputs = []
        num_lists = []
        seq_lens = []
        is_new_epoch = False
        if batch_size > self.data_size:
            print("Batch size must be smaller than data size")
            return

        for inc in range(batch_size):
            loc = self.pos + inc
            inputs.append(np.load(lines[loc][2]))# load feature
            num_lists.append(list(map(int, lines[loc][3].split(' ')))) #load
            seq_lens = append(int(lines[1][1])) # load sequence lens

        print(seq_lens)
        self.pos += batch_size
        labels = list2sparse(num_lists)

        # TODO: will not train on last several data points
        if self.pos + batch_size > self.data_size:
            self.pos = 0
            is_new_epoch = True

        return inputs, labels, seq_lens, is_new_epoch

