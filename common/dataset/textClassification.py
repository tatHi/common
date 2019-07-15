from common.dataset import dataset
class Dataset4TextClassification(dataset.Dataset):
    def __init__(self, path, useBEOS=False, charMode=False, vocab=None):
        super().__init__(path, useBEOS, charMode, vocab)
        self.labels = []
        self.labelSize = None
        self.label2index = {}
        self.__setLabelInfo()

    def __setLabelInfo(self):
        labels = set()
        for ty in self.data:
            for line in self.data[ty]:
                labels.add(line['label'])
        self.labels = sorted(list(labels)) # sorted
        self.labelSize = len(labels)

        # re-index labels because line['label'] may be string
        self.label2index = {label:i for i,label in enumerate(self.labels)}
        for ty in self.data:
            for line in self.data[ty]:
                line['label'] = self.label2index[line['label']]

def convertSentLabelFiles(sentDataPath, labelDataPath, outPath, splited=False):
    import os
    import json
    # convert 2 files (sentences.txt and labels.txt) into dataset-formatted file.json

    if '%s' not in sentDataPath and '%s' not in labelDataPath:
        print('error: there is no %s expression to embed train/(valid)/test.')
        print('the path must be like \"hoge_%s_text/label.txt\"')

    data = {}

    for ty in ['train','valid','test']:
        sp = sentDataPath%ty
        lp = labelDataPath%ty
        
        if not os.path.exists(sp) or not os.path.exists(lp):
            print('pass %s'%ty)
            continue
        
        sentData = [line.strip() for line in open(sp)]
        labelData = [line.strip() for line in open(lp)]

        if len(sentData) != len(labelData):        
            print('sizes of sentences and labels must be same. (sent:%d label:%d)'%(len(sentData),len(labelData)))
            print('sentDataPath: %s'%sp)
            print('labelDataPath: %s'%lp)
            exit()
        
        print('train:')
        print('\t%s(%d)'%(sp,len(sentData)))
        print('\t%s(%d)'%(lp,len(labelData)))
    
        tyData = []
        for s,l in zip(sentData, labelData):
            if splited:
                s = s.split()
            tyData.append({'text':s,
                           'label':l})

        data[ty] = tyData

    if data:
        with open(outPath,'w') as f:
            json.dump(data, f)
    else:
        print('exit: no data is dumped')
