import json
import numpy as np
from common.dataset import vocab

class Dataset:
    def __init__(self, path):
        self.data = json.load(open(path))
        self.vocab = vocab.Vocabulary(self.data['train'])

    def makeMiniBatchIdx(self, dataType, batchSize, shuffle=False):
        '''
        return indices for minibatch.
        note that this is not return data itself.
        '''
        assert dataType in self.data, 'dataType is not in keys('+list(self.data.keys())+')'
        indices = np.random.permutation(len(self.data[dataType])) 
        miniBatchIdx = pack(indices, batchSize)
        return miniBatchIdx

def pack(arr, size):
    batch = [arr[i:i+size] for i in range(0, len(arr), size)]
    return batch

    
