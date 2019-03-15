from glob import glob
from os.path import join

with open('../classes.txt','r') as fcl:
    classNames=[x.strip() for x in fcl.read().split('\n')]
    classNames=[x for x in classNames if x !='']

files={c:[] for c in classNames}

for folder in glob('?'):
    for c in classNames:
        with open(join(folder,'{0}.txt'.format(c))) as fin:
            files[c].extend(fin.read().split('\n'))

for c in classNames:
    with open('{0}.txt'.format(c),'w') as fout:
        fout.write('\n'.join(files[c]))
