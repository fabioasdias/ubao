from glob import glob
import sys
from os.path import join, basename
import numpy as np
from random import shuffle


if len(sys.argv)!=2:
    print('.py datadir')
    exit(-1)

datadir=sys.argv[1]

pTrain=0.6
pVal=0.35
pTest=0.05

xTrain=[]
yTrain=[]

xVal=[]
yVal=[]

xTest=[]
yTest=[]

panels=sorted(glob(join(sys.argv[1]+'*')))
with open('classes.txt','w') as fout:
    fout.write('\n'.join([basename(x) for x in panels]))
# exit()
p2i={p:i for i,p in enumerate([basename(x) for x in panels])}
i2p={p2i[p] for p in p2i.keys()}

for c in p2i.keys():
    i=p2i[c]
    print(c)
    curData=[]
    for fv in glob(join(datadir,c,'t_*.fv')):
        curData.append(np.loadtxt(fv))
    shuffle(curData)

    N=len(curData)

    nTest=int(np.ceil(pTest*N))
    xTest.extend(curData[:nTest])
    yTest.extend([i,]*nTest)

    nVal=int(np.ceil(pVal*N))
    xVal.extend(curData[nTest:(nTest+nVal)])
    yVal.extend([i,]*nVal)

    xTrain.extend(curData[(nTest+nVal):])
    yTrain.extend([i,]*(N-nTest-nVal))

print(len(yTrain),len(yVal),len(yTest))
# 16274 5471 5471

np.savez_compressed('panel_data.npz',
    xTrain=np.array(xTrain),
    yTrain=np.array(yTrain),
    xVal=np.array(xVal),
    yVal=np.array(yVal),
    xTest=np.array(xTest),
    yTest=np.array(yTest))