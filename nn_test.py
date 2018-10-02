import tensorflow as tf
from nn_model import *
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from shutil import copyfile
from os import makedirs
from os.path import join,basename,exists

tfd = tfp.distributions

# FLAGS=tf.flags

def main(argv):

    if len(argv)!=2:
        print('.py folder')
        exit(-1)

    folder=argv[1]

    with open('classes.txt','r') as fcl:
        classNames=[x.strip() for x in fcl.read().split('\n')]
        classNames=[x for x in classNames if x !='']

    right={}
    total={}

    for c in classNames:
        right[c]=0
        total[c]=0

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

            for p in glob(join(folder,'*')):
                for f in glob(join(p,'*.fv')):
                    X=np.loadtxt(f)
                    X = np.expand_dims(X, axis=0)
                    prediction = sess.run([predictions,], feed_dict={FV: X})[0]
                    guess=classNames[prediction[0]]
                    c=basename(p)
                    if (c==guess):
                        # print('+',guess,p,f)
                        right[c]+=1
                    # else:
                    #     print('-',guess,p,f)
                    total[c]+=1.0

    TR=0
    TT=0
    for c in sorted(total.keys()):
        TR+=right[c]
        TT+=total[c]
        print('{3} - {0:2.2f} ({1} / {2})'.format(100*(right[c]/total[c]),right[c],total[c],c))

    print('\n TOTAL {0:2.2f} ({1} / {2})'.format(100*(TR/TT),TR,TT))


if __name__ == "__main__":
  tf.app.run()
