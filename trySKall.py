from glob import glob
from os.path import join,basename, exists
import sys
import numpy as np
import matplotlib.pylab as plt
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.externals import joblib

if __name__ == '__main__':
    if len(sys.argv)!=3:
        print('.py train/ test/')
        exit(-1)
    

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]

    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
            "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
            "Naive Bayes", "QDA"]

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

    X=np.array(X)
    Y=np.array(Y)

    XT=[]
    YT=[]
    panelsTest=glob(join(sys.argv[2]+'*'))
    for p in panelsTest:
        for f in glob(join(p,'t_*.fv')):
            XT.append(np.loadtxt(f))
            YT.append(p2i[basename(p)])

    XT=np.array(XT)
    YT=np.array(YT)


    for name, clf in zip(names, classifiers):    
        clf.fit(X, Y)  
        # joblib.dump(clf, '_'+name)         
        print(name)
        YY=clf.predict(X)
        print('Train ',sum(YY==Y)/len(Y))
        YP=clf.predict(XT)
        print('TEST', sum(YP==YT)/len(YP))
        print("\n")