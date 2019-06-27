'''
json for input:
train:
    [{
        text: abcdefgh
        label: [3,2,3] //lenth
        label: [0,0,1,0,1,0,0,1] //01
    }]
'''

from common.dataset import dataset
class Dataset4WordSegmentation(dataset.Dataset):
    def __init__(self, path, labelType='length', useBEOS=False):
        if labelType not in ['length','01']:
            print('labelType should be length or 01')
            exit()
        
        super().__init__(path, useBEOS)

        # check
        for ty in self.data:
            for line in self.data[ty]:
                if len(line['text']) != sum(line['label']):
                    print(line['text'])
                    print(line['label'])
                    exit()

        if labelType=='length':
            self.maxLength = max([l for line in self.data['train'] for l in line['label']])
