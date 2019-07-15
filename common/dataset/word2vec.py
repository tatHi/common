from common.dataset import dataset
from common.lm import ngram
import numpy as np

class Dataset4Word2Vec(dataset.Dataset):
    def __init__(self, path, mode, windowSize=2,initVocab=None):
        if mode not in ['CBOW', 'SG']:
            print('Word2VecDataset\'s mode must be CBOW or SG')
            exit()

        self.mode = mode
        self.windowSize = windowSize
        
        super().__init__(path, useBEOS=True, charMode=False, initVocab=vocab)

        # discard datasets other than train-set
        self.idData = self.idData['train']
        
        # build unigram language model for negative sampling
        unigram = ngram.NgramLM(1, False)
        unigram.build(self.idData)
        self.nsDist = [unigram.getNgramProb(i)**(3/4) 
                       for i in range(self.vocab.vocabSize)]
        self.nsDist = np.array(self.nsDist)/sum(self.nsDist)

        # build dataset
        if 2<=self.windowSize: self.__repad();
        self.__buildDataset()

    def __repad(self):
        # re-padding BOS and EOS to match with windowSize
        for i,line in enumerate(self.idData):
            self.idData[i] = [line[0]]*self.windowSize \
                             + line[1:-1] \
                             + [line[-1]]*self.windowSize
            
    def __buildDataset(self):
        # only using training set
        self.trainSet = []
        
        '''
        trainSet contains (t, c, c, c, c)
        num of c depends on window size, e.g.=2, -2~+2
        
        if mode is CBOW:
            (t, (c-2, c-1, c+1, c+2))
        if mode is SG:
            (t, c-2)
            (t, c-1)
            (t, c+1)
            (t, c+2)
        '''
        
        for line in self.idData:
            for i in range(self.windowSize, len(line)-self.windowSize):
                target = line[i]
                contexts = [line[i+j] for j in range(-self.windowSize, self.windowSize+1) if j!=0]

                if self.mode=='CBOW':
                    pair = (target, contexts)
                    self.trainSet.append(pair)
                elif self.mode=='SG':
                    for c in contexts:
                        pair = (target, c)
                        self.trainSet.append(pair)

    def negativeSample(self, num):
        return np.random.choice(self.vocab.vocabSize, size=num, p=self.nsDist)

def test():
    import sys
    path = sys.argv[1]
    ds = Dataset4Word2Vec(path, mode='SG', windowSize=2)
    print(ds.negativeSample(100))

if __name__ == '__main__':
    test()
