from common.dataset import dataset
class Dataset4SentenceClassification(dataset.Dataset):
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
        
