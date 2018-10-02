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
from random import sample


if __name__ == '__main__':
    if len(sys.argv)!=2:
        print('.py data/')
        exit(-1)
    

    classifiers = [
#        KNeighborsClassifier(10),
        SVC(kernel="linear", C=0.05),
        SVC(gamma="scale", C=0.05),
#        GaussianProcessClassifier(1.0 * RBF(1.0)),
#        DecisionTreeClassifier(max_depth=10),
#        RandomForestClassifier(max_depth=10, n_estimators=10),
        MLPClassifier(alpha=1),
        # AdaBoostClassifier(),
        # GaussianNB(),
        #QuadraticDiscriminantAnalysis()
        ]

    names = [#"Nearest Neighbors", 
             "Linear SVM", 
             "RBF SVM",
             # "Gaussian Process",
 #           "Decision Tree", 
 #           "Random Forest", 
            "Neural Net", 
            #"AdaBoost",
            #"Naive Bayes", 
            #"QDA"
            ]

    XTrain=[]
    YTrain=[]
    XTest=[]
    YTest=[]


    panels=sorted(glob(join(sys.argv[1]+'*')))
    p2i={p:i for i,p in enumerate([basename(x) for x in panels])}
    i2p={p2i[p] for p in p2i.keys()}

    if exists('train.npz'):
        data=np.load('train.npz')
        XTest=data['XTest']
        YTest=data['YTest']
        XTrain=data['XTrain']
        YTrain=data['YTrain']
    else:
        for p in panels:
            cf=glob(join(p,'t_*.fv'))
            b=[basename(x) for x in cf]
            prefs=list(set([x[x.find('_'):x.rfind('_')+1] for x in b]))
            testnames=sample(prefs,int(np.ceil(0.2*len(prefs))))
            for f in cf:
                if any([(x in f) for x in testnames]):
                    XTest.append(np.loadtxt(f))
                    YTest.append(p2i[basename(p)])
                else:
                    XTrain.append(np.loadtxt(f))
                    YTrain.append(p2i[basename(p)])

        XTest=np.array(XTest)
        YTest=np.array(YTest)
        XTrain=np.array(XTrain)
        YTrain=np.array(YTrain)

        np.savez('train.npz', XTest=XTest, YTest=YTest, XTrain=XTrain, YTrain=YTrain)

    for name, clf in zip(names, classifiers):    
        clf.fit(XTrain, YTrain)  
        # joblib.dump(clf, '_'+name)         
        print(name)
        YPTrain=clf.predict(XTrain)
        print('Train ',sum(YTrain==YPTrain)/len(YTrain))
        YPTest=clf.predict(XTest)
        print('TEST', sum(YPTest==YTest)/len(YTest))
        print("\n")