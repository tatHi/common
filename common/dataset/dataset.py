import json
import numpy as np
from common.dataset import vocab

class Dataset:
    def __init__(self, path, useBEOS=False, useCharVocab=False, charMode=False, noUNK=False, initVocab=None, lengthOrder=False):
        '''
        useBEOS=True -> BOS t t t EOS
        charMode=True -> express a word as a sequence of characters like ['l','i','k','e']
        '''

        self.data = json.load(open(path))

        if lengthOrder:
            self.data = {k:[line for line in sorted(v, key=lambda x:len(x['text']), reverse=True)] for k,v in self.data.items()}

        if initVocab is None:
            self.vocab = vocab.Vocabulary(self.data['train'], noUNK=noUNK, useBEOS=useBEOS, charMode=False)
        else:
            self.vocab = initVocab

        self.charMode = charMode
        self.useCharVocab = charMode or useCharVocab
        if self.useCharVocab:
            self.charVocab = vocab.Vocabulary(self.data['train'], noUNK=noUNK, useBEOS=useBEOS, charMode=True)
        
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

    def makeMiniBatchIdx(self, dataType, batchSize, shuffle=False, lengthOrder=False, lengthLimit=-1):
        '''
        return indices for minibatch.
        note that this is not return data itself.

        lengthOrder: sort by their length in batch
        '''
        assert dataType in self.data, 'dataType is not in keys('+list(self.data.keys())+')'
        
        # shuffle
        if shuffle:
            indices = np.random.permutation(len(self.data[dataType])) 
        else:
            indices = np.arange(len(self.data[dataType]))

        # cut long training data
        if 0<=lengthLimit:
            indices = [idx for idx in indices if len(self.data[dataType][idx]['text'])<=lengthLimit]

        miniBatchIdx = pack(indices, batchSize)

        if lengthOrder:
            miniBatchIdx = [self.sortByUnitLength(batch, dataType) for batch in miniBatchIdx]
        return miniBatchIdx

    def sortByUnitLength(self, batch, dataType):
        neoBatch = [b for b in sorted(batch, key=lambda x:len(self.idData[dataType][x]))[::-1]]
        return neoBatch

    def unkDropout(self, dataType, idx, rate):
        idLine = self.idData[dataType][idx]
        mask = np.random.choice(2, len(idLine), p=[rate, 1-rate])
        idLine = idLine * mask
        idLine = idLine + np.logical_not(mask)*self.vocab.word2idDict['<UNK>']

        return idLine

def makeDatasetJSON(pathDict, saveName):
    '''
    pathDict = {'train':'/path',
                'test':'/path'}
    split sentence and pack them as json
    '''
    data = {key:[{'text':line.strip().split()} for line in open(pathDict[key])] for key in pathDict}
    with open(saveName, 'w') as f:
        json.dump(data, f, ensure_ascii=False)

    print('dump as json successfully')

def pack(arr, size):
    batch = [arr[i:i+size] for i in range(0, len(arr), size)]
    return batch

def test():
    import sys
    path = sys.argv[1]
    ds = Dataset(path, lengthOrder=True)

    #ds.makeMiniBatchIdx('train', 32, shuffle=False, lengthOrder=False, lengthLimit=20)
    ds.unkDropout('train', 99, 0.3)

if __name__ == '__main__':
    test()

