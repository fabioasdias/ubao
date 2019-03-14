from glob import glob
from os.path import join,basename, exists
import sys
import numpy as np
import matplotlib.pylab as plt
from sklearn import svm
from sklearn.externals import joblib

if __name__ == '__main__':
    if len(sys.argv)!=3:
        print('.py train/ test/')
        exit(-1)
    
    X=[]
    Y=[]

    panels=glob(join(sys.argv[1]+'*'))
    with open('./wtrain/classes.txt','r') as fcl:
        classNames=[x.strip() for x in fcl.read().split('\n')]
        classNames=[x for x in classNames if x !='']

    p2i={p:i for i,p in enumerate(classNames)}
    i2p={p2i[p] for p in p2i.keys()}
    for p in panels:
        for f in glob(join(p,'*.fv')):
            X.append(np.loadtxt(f))
            Y.append(p2i[basename(p)])
        # if (len(set(Y))>2):
        #     break


    X=np.array(X)
    print(X.shape)
    Y=np.array(Y)
    print(Y.shape)
    svmFile='svm.model'
    if (exists(svmFile)):
        clf = joblib.load(svmFile) 
        print('loaded')
    else:
        clf = svm.SVC(gamma='scale',class_weight='balanced')
        clf.fit(X, Y)  
        joblib.dump(clf, svmFile)         

    YY=clf.predict(X)
    print('Train ',sum(YY==Y)/len(Y))


    XT=[]
    YT=[]
    panelsTest=glob(join(sys.argv[2]+'*'))
    for p in panelsTest:
        # if (p2i[basename(p)] not in Y):
        #     continue
        for f in glob(join(p,'t_*.fv')):
            XT.append(np.loadtxt(f))
            YT.append(p2i[basename(p)])

    XT=np.array(XT)
    YT=np.array(YT)
    print(XT.shape)
    print(YT.shape)
    YP=clf.predict(XT)
    print(YP.shape)
    print(sum(YP==YT)/len(YP))