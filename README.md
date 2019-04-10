# common
じぶんよう.  
python3.  
```
$ git clone https://github.com/tathi/common
$ cd common
$ python setup.py develop
```

MeCabを入れるのが面倒な環境なら、setup.pyの以下から`mecab-python3`を削除する。
```
install_requires=['mecab-python3','numpy']
```
