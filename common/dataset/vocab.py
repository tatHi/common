class Vocabulary:
    def __init__(self, data):
        self.word2idDict = {}
        self.id2wordDict = {}
        self.wordCountDict = {}
        self.wordNum = 0

        self.__buildVocabulary(data) # dataset.Dataset.data['train']

    def __buildVocabulary(self, data):
        # count
        for line in data:
            text = line['text']
            for word in text:
                if word in self.wordCountDict:
                    self.wordCountDict[word] += 1
                else:
                    self.wordCountDict[word] = 1
                self.wordNum += 1

        # UNK
        self.wordCountDict['<UNK>'] = 1
        self.wordNum += 1

        # indexing
        for word, num in sorted(self.wordCountDict.items(), key=lambda x:x[1])[::-1]:
            self.word2idDict[word] = len(self.word2idDict)
            self.id2wordDict[self.word2idDict[word]] = word

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
