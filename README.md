# tf_rnn_tutorial
TensorFlow tutorial for recurrent neural network

# Tutorial 1: Minimal RNN language model

This model learns from a plane text and produces pseudo-text.
The tutorial uses the following TensorFlow functions.
 - tf.nn.dynamic_rnn()
 - tf.contrib.rnn.BasicLSTMCell()
This example shows how to feed 'initial_state' of tf.nn.dynamic_rnn() through TensorFlow's placeholders.
