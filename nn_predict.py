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

    if len(argv)==2:
        doCopy=False
    elif len(argv)==3:
        doCopy=True
        toFolder=argv[2]
    else:
        print('.py originFolder [copy to folder]')
        exit(-1)

    folder=argv[1]

    with open('./wtrain/classes.txt','r') as fcl:
        classNames=[x.strip() for x in fcl.read().split('\n')]
        classNames=[x for x in classNames if x !='']

    

    with tf.Graph().as_default():
    # Build a Bayesian LeNet5 network. We use the Flipout Monte Carlo estimator
    # for the convolution and fully-connected layers: this enables lower
    # variance stochastic gradients than naive reparameterization.
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
                guess=classNames[prediction[0]]
                print(f,guess)
                if doCopy:
                    copyto=join(toFolder,guess)
                    if (not exists(copyto)):
                        makedirs(copyto)
                    imgFile=f.replace('.fv','.jpg')
                    copyfile(imgFile,join(copyto,basename(imgFile)))


if __name__ == "__main__":
  tf.app.run()
