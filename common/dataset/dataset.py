import json
import numpy as np
from common.dataset import vocab

class Dataset:
    def __init__(self, path, useBEOS=False, charMode=False):
        '''
        useBEOS=True -> BOS t t t EOS
        charMode=True -> express a word as a sequence of characters like ['l','i','k','e']
        '''

        self.data = json.load(open(path))
        self.vocab = vocab.Vocabulary(self.data['train'], useBEOS, charMode=False)
        
        self.charMode = charMode
        if charMode:
            self.charVocab = vocab.Vocabulary(self.data['train'], useBEOS, charMode=True)
        
        self.__buildDataset(useBEOS, charMode)

    def __buildDataset(self, useBEOS, charMode):
        self.idData = {}
        for dt in self.data:
            self.idData[dt] = []
            for line in self.data[dt]:
                if charMode:
                    idLine = [self.charVocab.words2ids(word) for word in line['text']]
                else:
                    idLine = self.vocab.words2ids(line['text'])
                
                if useBEOS:
                    if charMode:
                        idLine = [[self.vocab.word2id('<BOS>')]] + idLine + [[self.vocab.word2id('<EOS>')]]
                    else:
                        idLine = [self.vocab.word2id('<BOS>')] + idLine + [self.vocab.word2id('<EOS>')]
                self.idData[dt].append(idLine)

    def makeMiniBatchIdx(self, dataType, batchSize, shuffle=False, lengthOrder=False):
        '''
        return indices for minibatch.
        note that this is not return data itself.
        '''
        assert dataType in self.data, 'dataType is not in keys('+list(self.data.keys())+')'
        if shuffle:
            indices = np.random.permutation(len(self.data[dataType])) 
        else:
            indices = np.arange(len(self.data[dataType]))
        miniBatchIdx = pack(indices, batchSize)

        if lengthOrder:
            miniBatchIdx = [self.sortByUnitLength(batch, dataType) for batch in miniBatchIdx]
        return miniBatchIdx

    def sortByUnitLength(self, batch, dataType):
        neoBatch = [b for b in sorted(batch, key=lambda x:len(self.idData[dataType][x]))[::-1]]
        return neoBatch

def pack(arr, size):
    batch = [arr[i:i+size] for i in range(0, len(arr), size)]
    return batch

def test():
    import sys
    path = sys.argv[1]
    ds = Dataset(path, True, True)
    print(ds.idData['train'][0])
    print(ds.data['train'][0]['text'])

if __name__ == '__main__':
    test()

