from common.dataset import dataset
class Dataset4TextClassification(dataset.Dataset):
    def __init__(self, path):
        super().__init__(path)
        self.labels = []
        self.labelSize = None
        self.__setLabelInfo()

    def __setLabelInfo(self):
        labels = set()
        for ty in self.data:
            for line in self.data[ty]:
                labels.add(line['label'])
        self.labels = sorted(list(labels))
        self.labelSize = len(labels)

def convertSentLabelFiles(sentDataPath, labelDataPath, outPath):
    # convert 2 files (sentences.txt and labels.txt) into dataset-formatted file.json

    if '%s' is not in sentDataPath and '%s' is not in labelDataPath:
        print('error:')

    sentData = [line.strip() for line in open(sentDataPath)]
    labelData = [line.strip() for line in open(labelDataPath)]

    if len(sentData) != len(labelData):        
        print('sizes of sentences and labels must be same. (sent:%d label:%d)'%(len(sentData),len(labelData)))
        exit()

    data = []
    for s,l in zip(sentData, labelData):
        line = {'text':s, 'label':l}
