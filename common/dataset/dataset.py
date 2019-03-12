import json
import numpy as np
from common.dataset import vocab

class Dataset:
    def __init__(self, path, useBEOS=False):
        self.data = json.load(open(path))
        self.vocab = vocab.Vocabulary(self.data['train'], useBEOS)
        self.__buildDataset(useBEOS)

    def __buildDataset(self, useBEOS):
        self.idData = {}
        for dt in self.data:
            self.idData[dt] = []
            for line in self.data[dt]:
                idLine = self.vocab.words2ids(line['text'])
                if useBEOS:
                    idLine = [self.vocab.word2id('<BOS>')] + idLine + [self.vocab.word2id('<EOS>')]
                self.idData[dt].append(idLine)

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

def test():
    path = '/Users/tatsuya-h/work/data/twitter_en_text.json'
    ds = Dataset(path, True)

if __name__ == '__main__':
    test()
