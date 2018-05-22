import csv
import numpy as np

def padInputs(inputs, seq_lens, batch_size):
    max_len = max(seq_lens)
    num_feature = inputs[0].shape[1]
    # default is np.float64
    result = np.zeros((max_len, batch_size, num_feature), dtype=np.float64)
    for idx, data in enumerate(inputs):
        result[0:data.shape[0],idx,:] = data
    #print(result)
    return result
    
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
    def __init__(self, mode='train', batch_size=1):
        super(DataProvider, self).__init__()

        # set csv_path and batch_size
        if mode == 'train':
            self.csv_path = "/data/ASR/corpus/librispeech/dataset/numpy/100h/train/character.csv"
        if mode == 'dev':
            self.csv_path = "/data/ASR/corpus/librispeech/dataset/numpy/100h/dev_clean/character.csv"
        if mode == 'test':
            self.csv_path = "/data/ASR/corpus/librispeech/dataset/numpy/100h/test_clean/character.csv"
        self.batch_size = batch_size 

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

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def next(self):
        inputs = []
        num_lists = []
        seq_lens = []
        is_new_epoch = False
        if self.batch_size > self.data_size:
            print("Batch size must be smaller than data size")
            return

        for inc in range(self.batch_size):
            line = self.lines[self.pos + inc]
            #print(np.load(line[2]).shape)
            #print(np.load(line[2]).dtype)
            inputs.append(np.load(line[2]))# load feature
            num_lists.append(list(map(int, line[3].split(' ')))) #load
            seq_lens.append(int(line[1])) # load sequence lens

        #print(seq_lens)
        self.pos += self.batch_size

        inputs = padInputs(inputs, seq_lens, self.batch_size)
        labels = list2sparse(num_lists)

        # TODO: will not train on last several data points
        if self.pos + self.batch_size > self.data_size:
            self.pos = 0
            is_new_epoch = True
        return inputs, labels, seq_lens, is_new_epoch

