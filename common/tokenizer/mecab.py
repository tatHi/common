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
    segedData = [tagger.parse(line) for line in open(filePath)]
    
    with open(resultPath,'w') as f:
        for line in segedData:
            f.write(line)
    print('done')
