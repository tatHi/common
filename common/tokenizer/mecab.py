import MeCab
def tokenize(line):
    tagger = MeCab.Tagger('-Owakati')
    segedLine = tagger.parse(line).strip().split()
    return segedLine

def tokenizeData(data):
    tagger = MeCab.Tagger('-Owakati')
    segedData = [tagger.parse(line).strip().split() for line in data]
    return segedData

def makeTokenizedFile(filePath):
    resultPath = filePath.replace('.','_mecab.')
    print('input file:', filePath)
    print('output file:', resultPath)
    
    tagger = MeCab.Tagger('-Owakati')
    data = [line.strip() for line in open(filePath)]
    segedData = [tagger.parse(line).strip() for line in data]

    with open(resultPath,'w') as f:
        for line in segedData:
            f.write(line+'\n')
    print('done')

