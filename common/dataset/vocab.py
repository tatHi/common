class Vocabulary:
    def __init__(self, data, useBEOS, charMode=False):
        self.word2idDict = {}
        self.id2wordDict = {}
        self.wordCountDict = {}
        self.wordNum = 0
        self.__buildVocabulary(data, useBEOS, charMode) # dataset.Dataset.data['train']

    def __buildVocabulary(self, data, useBEOS, charMode):
        # count
        for line in data:
            text = line['text']

            if charMode:
                # expand characters
                text = [c for word in text for c in word]

            for word in text:
                if word in self.wordCountDict:
                    self.wordCountDict[word] += 1
                else:
                    self.wordCountDict[word] = 1
                self.wordNum += 1

        # UNK
        self.wordCountDict['<UNK>'] = 1
        self.wordNum += 1

        # BOS, EOS
        if useBEOS:
            for w in ['<BOS>','<EOS>']:
                self.wordCountDict[w] = len(data)
                self.wordNum += len(data)

        # indexing
        for word, num in sorted(self.wordCountDict.items(), key=lambda x:x[1])[::-1]:
            self.addWord(word, updateVocabSize=False)

        # vocab size
        self.vocabSize = len(self.word2idDict)

    def addWord(self, word, updateVocabSize=True):
        self.word2idDict[word] = len(self.word2idDict)
        self.id2wordDict[self.word2idDict[word]] = word

        if updateVocabSize:
            self.vocabSize = len(self.word2idDict)

    def word2id(self, word):
        if word in self.word2idDict:
            return self.word2idDict[word]
        else:
            return self.word2idDict['<UNK>']

    def id2word(self, idx):
        assert idx in self.id2wordDict, 'access to undefined idx in vocabulary.'
        return self.id2wordDict[idx]

    def words2ids(self, words):
        ids = [self.word2id(word) for word in words]
        return ids

    def ids2words(self, ids):
        words = [self.id2word(i) for i in ids]
        return words
