class Dataset:
    def __init__(self, pathes, split=True):
        '''
        get pathes, set data
        pathes: [trainDataPath, validDataPath, testDataPath, ...]
        split=True -> split lines with whitespace
        '''
        self.data = [[line.strip().split() if split else line.strip() 
                            for line in open(path)] for path in pathes]
        
