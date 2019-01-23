import MeCab
def tokenize(line):
    tagger = MeCab.Tagger('-Owakati')
    segedLine = tagger.parse(line).strip().split()
    return segedLine

def tokenizeData(data):
    segedData = [tokenize(line) for line in data]
    return segedData

