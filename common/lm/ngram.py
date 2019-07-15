import numpy as np
class NgramLM:
    def __init__(self, n, useMgram=True):
        # data is list of list
        self.n = n
        self.useMgram = useMgram # set n-1 gram automatically or not
        if n>=2 and self.useMgram:
            self.mgramLM = NgramLM(n-1) # m=n-1
        
        self.ngramCountDict = {}
        self.numOfNgram = 0

    def build(self, data):
        for line in data:
            self.addLine(line)
    
    def addNgram(self, ngram):
        if ngram in self.ngramCountDict:
            self.ngramCountDict[ngram] += 1
        else:
            self.ngramCountDict[ngram] = 1
        self.numOfNgram += 1

    def reduceNgram(self, ngram):
        self.ngramCountDict[ngram] -= 1
        self.numOfNgram -= 1
        
        if self.ngramCountDict[ngram]==0:
            del self.ngramCountDict[ngram]

    def addLine(self, line):
        if len(line)<self.n:
            return
        for i in range(len(line)-self.n+1):
            if self.n==1:
                ngram = line[i]
            else:
                ngram = tuple(line[i:i+self.n])
            self.addNgram(ngram)
        if self.n>=2 and self.useMgram:
            self.mgramLM.addLine(line)

    def reduceLine(self, line):
        if len(line)<self.n:
            return
        for i in range(len(line)-self.n+1):
            if self.n==1:
                ngram = line[i]
            else:
                ngram = tuple(line[i:i+self.n])
            self.reduceNgram(ngram)
        if self.n>=2 and self.useMgram:
            self.mgramLM.reduceLline(line)

    def getCount(self, ngram):
        if (type(ngram)==list or type(ngram)==tuple) and len(ngram)==1:
            ngram = ngram[0]
        if self.n >= 2:
            ngram = tuple(ngram)
        return self.ngramCountDict[ngram] if ngram in self.ngramCountDict else 0

    def getNgramProb(self, ngram):
        assert self.n==1 or (self.n>=2 and self.useMgram), 'build LM with useMgram=True to get ngram probability'
        if self.n==1:
            return self.getCount(ngram)/self.numOfNgram
        else:
            if self.n==2:
                mgram = ngram[0]
            else:
                mgram = ngram[:-1]
            return self.getCount(ngram)/self.mgramLM.getCount(mgram)

    def getLineProb(self, line):
        assert self.n==1 or (self.n>=2 and self.useMgram), 'build LM with useMgram=True to get ngram probability'
        if self.n==1:
            p = np.prod([self.getNgramProb(ngram) for ngram in line])
        else:
            p = self.mgramLM.getLineProb(line[:self.n-1])
            for i in range(len(line)-self.n+1):
                ngram = line[i:i+self.n]
                mgram = ngram[:-1] if self.n>=3 else ngram[0]
                p *= self.getCount(ngram)/self.mgramLM.getCount(mgram)
        return p

if __name__ == '__main__':
    b = NgramLM(2)
    data = [['a','b'],['b','c','d']]
    b.build(data)
    print(b.ngramCountDict)
    print(b.numOfNgram)
    print(b.mgramLM.ngramCountDict)
    print(b.mgramLM.numOfNgram)
    print(b.getLineProb(['a','b','c']))
