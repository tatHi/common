from common.dataset import dataset
class Dataset4NER(dataset.Dataset):
    def __init__(self, path, useBEOS=False):
        super().__init__(path, useBEOS)
        self.labels = []
        self.labelSize = None
        self.label2index = {}
        self.__setLabelInfo()

    def __setLabelInfo(self):
        labels = set()
        for ty in self.data:
            for line in self.data[ty]:
                labels |= set(line['label'])
        self.labels = sorted(list(labels)) # sorted
        self.labelSize = len(labels)

        # re-index labels because labels in line['label'] may be string
        self.label2index = {label:i for i,label in enumerate(self.labels)}
        self.index2label = {v:k for k, v in self.label2index.items()}
        for ty in self.data:
            for line in self.data[ty]:
                line['label'] = [self.label2index[label] for label in line['label']]
