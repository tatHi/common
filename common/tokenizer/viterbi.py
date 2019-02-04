class ViterbiNode:
    def __init__(self, score, token):
        self.score = score
        self.token = token

# unigram
val = 1e15
def viterbiTokenize(line, scoreDict, maxLength, mode='min'):
    assert mode in ['min','max'], 'mode error: mode should be max of min (String)'

    # forward
    nodes = []
    for i in range(len(line)):
        node = ViterbiNode(val,None) if mode=='min' else ViterbiNode(-val,None)
        for l in range(min(maxLength,i+1)):
            token = line[i-l:i+1]
            score = scoreDict[token] if token in scoreDict else (val if mode=='min' else -val)
            if (mode=='min' and score<node.score) or (mode=='max' and node.score<score):
                node.score=score
                node.token=token
        nodes.append(node)

    # backward
    tokens = []
    j = len(line)-1
    while 0<=j:
        node = nodes[j]
        tokens.append(node.token)
        j -= len(node.token)

    return tokens[::-1]

if __name__ == '__main__':
    d = {'a':1, 'b':1, 'c':1, 'ab':1, 'bc':2,}
    line = 'abc'
    maxLength = 2
    print(viterbiTokenize(line,d,maxLength,mode='min'))
