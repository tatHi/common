import dataset
class Dataset4SentenceClassification(dataset.Dataset):
    def __init__(self, textPathes, labelPathes, split=True):
        super().__init__(textPathes,split)
        self.labels = [[line.strip() for line in open(path)] for path in labelPathes]

