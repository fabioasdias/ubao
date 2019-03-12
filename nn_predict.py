import tensorflow as tf
from nn_model import *
import tensorflow_probability as tfp
# import matplotlib.pyplot as plt
from shutil import copyfile
from os import makedirs
from os.path import join,basename,exists
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# from sklearn.manifold import MDS
import json
from tqdm import tqdm
import pickle

tfd = tfp.distributions
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)

cachefile='cache.gp'

# FLAGS=tf.flags

# def saveStuff(FVs,fnames):
#     print('saving')
#     for c in fnames:
#         print(c)
#         f=c+'_'+cachefile

#         X=np.array(np.squeeze(FVs[c]))
#         np.savez_compressed(f.replace('.gp','.npz'),FVs=X)

#         with open(f,'wb') as fgp:
#             pickle.dump({'fnames':fnames[c]},fgp)
#     with open(cachefile,'wb') as fgp:
#         pickle.dump({'classNames':list(fnames.keys())})

# def readStuff(c):
#     f=c+'_'+cachefile
#     with open(f,'rb') as fgp:
#         fnames=pickle.load(fgp)['fnames']
#     FVs=np.load(f.replace('.gp','.npz'))['FVs']
#     return({'FVs':FVs,'fnames':fnames})

def main(argv):

    if len(argv)!=2:
        print('.py filelist.txt')
        exit(-1)

    fileList=argv[1]

    with open('../classes.txt','r') as fcl:
        classNames=[x.strip() for x in fcl.read().split('\n')]
        classNames=[x for x in classNames if x !='']

    NBATCH=1000

    if not exists(cachefile):
        # FVs={}
        fnames={}
        for c in classNames:
            # FVs[c]=[]
            fnames[c]=[]

        with tf.Graph().as_default():
            FV = tf.keras.Input(shape=[4096,])
            with tf.name_scope("bayesian_neural_net", values=[FV]):
                neural_net=get_model()
                logits = neural_net(FV)
                predictions = tf.argmax(logits, axis=1)


            with tf.name_scope("predict"):
                sess = tf.Session()                   
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                tf.saved_model.loader.load(sess, ["trained"], FLAGS.model_dir)

                with open(fileList) as flist:
                    todo=[x.strip().replace('.jpg','.fv') for x in flist.read().split('\n') if x!='']
                todo=[todo[x:x+NBATCH] for x in range(0, len(todo), NBATCH)]                        
                for lf in tqdm(todo):
                    X=[]
                    for f in lf:
                        X.append(np.loadtxt(f))
                    X=np.array(X)
                    prediction = sess.run([predictions,], feed_dict={FV: X})[0]
                    for i in range(NBATCH):
                        c=classNames[prediction[i]]
                        fnames[c].append(basename(lf[i]).replace('.fv','.jpg'))
        # saveStuff(FVs,fnames)
        # del(FVs)           
        # del(fnames)
        
        
    for c in tqdm(classNames):
        # data=readStuff(c)
        # X=np.squeeze(np.array(data['FVs']))
        # del(FVs[c])
        # print(X.shape)
        # M=MDS(n_components=2, metric=False).fit(X[:500,:])
        # Y = M.transform(X)
        # Y = PCA(n_components=2).fit_transform(X)
        # Y=scaler.fit_transform(Y)
        # ymin=np.min(Y,axis=0)
        # ymax=np.max(Y,axis=0)        
        # Y=(Y-ymin)/(ymax-ymin)

        with open(c+'.txt','w') as fout:
            fout.write('\n'.join(fnames[c]))

        # imglist=[ {'left':Y[i,0], 'top':Y[i,1], 'fname':name} for i,name in enumerate(data['fnames'])]        
        # with open(c+'.json','w') as fout:
            # json.dump(imglist,fout)
        # print(c,len(imglist))

    master=[{'name':c,'fname':c+'.json'} for c in classNames]
    with open('master.json','w') as fout:
        json.dump(master,fout)


if __name__ == "__main__":
  tf.app.run()
