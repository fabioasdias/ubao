import tensorflow as tf
from nn_model import *
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from shutil import copyfile
from os import makedirs
from os.path import join,basename,exists
from sklearn.decomposition import PCA
import json

tfd = tfp.distributions

# FLAGS=tf.flags

def main(argv):

    if len(argv)==2:
        doCopy=False
    elif len(argv)==3:
        doCopy=True
        toFolder=argv[2]
    else:
        print('.py originFolder [copy to folder-not working]')
        exit(-1)

    folder=argv[1]

    with open('classes.txt','r') as fcl:
        classNames=[x.strip() for x in fcl.read().split('\n')]
        classNames=[x for x in classNames if x !='']

    
    FVs={}
    fnames={}
    for c in classNames:
        FVs[c]=[]
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

            for f in glob(os.path.join(folder,'*.fv')):
                X=np.loadtxt(f)
                X = np.expand_dims(X, axis=0)
                prediction = sess.run([predictions,], feed_dict={FV: X})[0]
                c=classNames[prediction[0]]
                FVs[c].append(X)
                fnames[c].append(basename(f).replace('.fv','.jpg'))
                # print(f,guess)
                # if doCopy:
                #     copyto=join(toFolder,guess)
                #     if (not exists(copyto)):
                #         makedirs(copyto)
                #     imgFile=f.replace('.fv','.jpg')
                #     target=join(copyto,basename(imgFile))
                #     copyfile(imgFile,target)
                #     copyfile(f,target.replace('.jpg','.fv'))


    for c in classNames:
        X=np.squeeze(np.array(FVs[c]))
        # print(X.shape)
        Y=PCA(n_components=2,whiten=True).fit_transform(X)
        # print(Y.shape,np.min(Y),np.max(Y))
        ymin=np.min(Y,axis=0)
        ymax=np.max(Y,axis=0)
        Y=np.sqrt((Y-ymin)/(ymax-ymin))
        imglist=[ {'left':Y[i,0], 'top':Y[i,1], 'fname':name} for i,name in enumerate(fnames[c])]        
        with open(c+'.json','w') as fout:
            json.dump(imglist,fout)
        print(c,len(imglist))
    master=[{'name':c,'fname':c+'.json'} for c in classNames]
    with open('master.json','w') as fout:
        json.dump(master,fout)



if __name__ == "__main__":
  tf.app.run()
