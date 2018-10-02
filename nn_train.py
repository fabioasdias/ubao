import tensorflow as tf
from nn_model import *
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
tfd = tfp.distributions


def main(argv):
  del argv  # unused

  xlog=[]
  tlog=[]
  vlog=[]

  if tf.gfile.Exists(FLAGS.model_dir):
    tf.logging.warning(
        "Warning: deleting old log directory at {}".format(FLAGS.model_dir))
    tf.gfile.DeleteRecursively(FLAGS.model_dir)
  # tf.gfile.MakeDirs(FLAGS.model_dir)

  panel_data = read_panels_fv(FLAGS.data_file)

  with tf.Graph().as_default():
    (fvs, labels, handle,
     training_iterator, heldout_iterator, test_iterator) = build_input_pipeline(
         panel_data, FLAGS.batch_size, panel_data['yVal'].shape[0])

    # Build a Bayesian LeNet5 network. We use the Flipout Monte Carlo estimator
    # for the convolution and fully-connected layers: this enables lower
    # variance stochastic gradients than naive reparameterization.
    with tf.name_scope("bayesian_neural_net", values=[fvs]):
      neural_net = get_model()
      logits = neural_net(fvs)
      labels_distribution = tfd.Categorical(logits=logits)

    # Compute the -ELBO as the loss, averaged over the batch size.
    neg_log_likelihood = -tf.reduce_mean(labels_distribution.log_prob(labels))
    kl = sum(neural_net.losses) / panel_data['yTest'].shape[0]
    elbo_loss = neg_log_likelihood + kl

    # Build metrics for evaluation. Predictions are formed from a single forward
    # pass of the probabilistic layers. They are cheap but noisy predictions.
    predictions = tf.argmax(logits, axis=1)
    accuracy, accuracy_update_op = tf.metrics.accuracy(
        labels=labels, predictions=predictions)

    # Extract weight posterior statistics for layers with weight distributions
    # for later visualization.
    names = []
    qmeans = []
    qstds = []
    for i, layer in enumerate(neural_net.layers):
      try:
        q = layer.kernel_posterior
      except AttributeError:
        continue
      names.append("Layer {}".format(i))
      qmeans.append(q.mean())
      qstds.append(q.stddev())


    with tf.name_scope("train"):
        opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        train_op = opt.minimize(elbo_loss)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # Run the training loop.
        train_handle = sess.run(training_iterator.string_handle())
        heldout_handle = sess.run(heldout_iterator.string_handle())
        test_handle = sess.run(test_iterator.string_handle())


        for step in range(FLAGS.max_steps):
          _ = sess.run([train_op, accuracy_update_op],
                      feed_dict={handle: train_handle})

          if step % 1000 == 0:
            loss_value, accuracy_value = sess.run(
                [elbo_loss, accuracy], feed_dict={handle: train_handle})
            print("\nStep: {:>3d} Loss: {:.3f} Accuracy: {:.3f}".format(
                step, loss_value, accuracy_value))
            xlog.append(step)
            tlog.append(accuracy_value)
            loss_value, accuracy_value = sess.run(
                [elbo_loss, accuracy], feed_dict={handle: heldout_handle})
            print("Step: {:>3d} Loss: {:.3f} Accuracy: {:.3f}".format(
                step, loss_value, accuracy_value))
            vlog.append(accuracy_value)
            plt.figure()
            plt.plot(xlog,tlog,'b')
            plt.plot(xlog,vlog,'r')
            plt.legend(['Train','Validation'])
            plt.savefig('curves.png',dpi=300)
            plt.close()
            
        builder = tf.saved_model.builder.SavedModelBuilder(FLAGS.model_dir)
        builder.add_meta_graph_and_variables(sess,
                            ["trained"],
                            strip_default_attrs=True)

        builder.save()
          # if  ( step == (FLAGS.max_steps-1)):# or ((step % 1000) == 0):
          #   builder.add_meta_graph_and_variables(sess,
          #                              ["trained"],
          #                              strip_default_attrs=True)
          #   builder.save()
          # if  ( step == (FLAGS.max_steps-1)) or ((step % 1000) == 0):            
          #   neural_net.save_weights(os.path.join(FLAGS.model_dir,'w{0}.h5'.format(step)))


    loss_value, accuracy_value = sess.run([elbo_loss, accuracy], feed_dict={handle: test_handle})
    print("TEST Loss: {:.3f} Accuracy: {:.3f}".format(loss_value, accuracy_value))


if __name__ == "__main__":
  tf.app.run()
