from glob import glob
from os.path import join
import sys
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pylab as plt
from sklearn.decomposition import PCA

if __name__ == '__main__':
    if len(sys.argv)!=2:
        print('.py folder')
        exit(-1)
    
    X=[]
    Y=[]

    panels=glob(join(sys.argv[1]+'*'))
    p2i={p:i for i,p in enumerate(panels)}
    for p in panels:
        for f in glob(join(p,'im*.fv')):
            X.append(np.loadtxt(f))
            Y.append(p2i[p])


    X=np.array(X)
    Y=np.array(Y)

    print(X.shape)
    cmap=plt.get_cmap('tab20')
    # X_embedded = TSNE(n_components=2).fit_transform(X)
    X_embedded = PCA(n_components=2).fit_transform(X)
    print(X_embedded.shape)
    
    c=0
    plt.figure()
    for i in set(Y):
        inds= np.where(Y==i)
        plt.plot(X_embedded[inds,0],X_embedded[inds,1],'.',c=cmap(c))
        c+=1
        if (c>10):
            plt.figure()
            c=0

    plt.show()


