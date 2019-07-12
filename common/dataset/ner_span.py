from common.dataset import dataset
class Dataset4NER_span(dataset.Dataset):
    def __init__(self, path, useBEOS=False, charMode=False, maxLength=3):
        super().__init__(path, useBEOS, charMode)
        self.maxLength = maxLength
        self.labels = []
        self.labelSize = None
        self.label2index = {}
        self.__setLabelInfo()
        
    def __setLabelInfo(self):
        labels = set()
        for ty in self.data:
            for line in self.data[ty]:
                labels |= set(line['label'])

        # normalize
        labels = {label if label=='O' else label.split('-')[1] for label in labels}
        self.labels = sorted(list(labels)) # sort list for reproductivity
        self.labelSize = len(labels)

        # re-index labels because labels in line['label'] may be string
        self.label2index = {label:i for i,label in enumerate(self.labels)}
        self.index2label = {v:k for k, v in self.label2index.items()}

        # set labeo info under the maxLength limitation
        for ty in self.data:
            for line in self.data[ty]:
                nes = extractNEs(line['label'])
                line['label'] = self.getSpanLabel(line['text'], nes)

    def getSpanLabel(self, line, nes):
        labels = []

        nes = nes[::-1] # reverse to use pop

        for i in range(len(line)):
            for j in range(min(self.maxLength, len(line)-i)):
                if nes and i==nes[-1][1] and i+j+1==nes[-1][2]:
                    labels.append(self.label2index[nes[-1][0]])
                    nes.pop()
                else:
                    labels.append(self.label2index['O'])

        return labels
    
def extractNEs(labels):
    nes = []
    ne = None
    for i,label in enumerate(labels):
        if label.startswith('B-'):
            ne = [label.split('-')[1], i, None]
        if ne and label=='O':
            ne[2] = i
            nes.append(ne)
            ne = None

    if ne:
        ne[2] = len(labels)
        nes.append(ne)

    return nes

if __name__ == '__main__':
    import sys
    ds = Dataset4NER_span(sys.argv[1])
    for line in (ds.data['train'][:10]):
        print(line['text'])
        n = 0
        for i in range(len(line['text'])):
            for j in range(min(ds.maxLength, len(line['text'])-i)):
                print(line['label'][n], end=',')
                n += 1
            print('')
