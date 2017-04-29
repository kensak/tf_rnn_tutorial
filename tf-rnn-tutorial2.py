"""
Simple character-level RNN model using TensorFlow.
Tutorial 2 - Using Summary

Copyright (c) Tsuyoshi Matsumoto 2017.

Dual licensed under the MIT license http://www.opensource.org/licenses/mit-license.php
and GPL v2 license http://www.gnu.org/licenses/gpl.html .

------------------------------------------------------------------------------------

TensorBoard allows us to visualize our TensorFlow graph, plot quantitative metrics about
the execution of your graph, and show additional data like images that pass through it.
https://www.tensorflow.org/get_started/summaries_and_tensorboard

To use TensorBoard, you need to write your data into files using summary writer.
This tutorial explains how to do it.

Follow these steps.
Step 1: in your TensorFlow graph, annotate
        nodes with summary operations.
Step 2: merge summaries.
        Combine all the summary nodes you've created
        into a single op that generates all the summary
        data.
Step 3: create a summary writer.
Step 4: evaluate your merged summary node.
Step 5: write the evaluated summary into the file.
Step 6: close the summary writer.

After the calculation, run the following command.
tensorboard --logdir=log
Then you can navigate to the displayed URL to visualize your execution.
"""
import argparse
import sys
import tensorflow as tf
import numpy as np
import math

# hyperparameters
B = 20  #batch size
H = 200 # size of hidden layer of neurons
T = 25  # number of time steps to unroll the RNN for
max_grad_norm = 5 # for gradient clipping

#----------------
# parse arguments
#----------------
parser = argparse.ArgumentParser()
parser.add_argument('--max_epoch', type=int, default=20000,
                    help='Number of epochs to run trainer.')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='Learning rate')
parser.add_argument('--data_file', type=str,
                    default='data/KoizumiYagumo_all.txt',
                    help='Directory for storing input data')
parser.add_argument('--log_dir', type=str, default='log',
                    help='Summaries log directory')
FLAGS, unparsed = parser.parse_known_args()

#----------------
# prepare data
#----------------

# should be simple plain text file
data = open(FLAGS.data_file, 'r').read()
chars = sorted(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has {0} characters, {1} unique.'.format(data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

input_index_raw = np.array([char_to_ix[ch] for ch in data])
input_index_raw = input_index_raw[0:len(input_index_raw) // T * T]
# Shift input_index_raw by one.
input_index_raw_shift = np.append(input_index_raw[1:], input_index_raw[0])
input_all = input_index_raw.reshape([-1, T])
target_all = input_index_raw_shift.reshape([-1, T])
num_batch = len(input_all)

#----------------
# build model
#----------------

class Model(object):
  """The RNN language model."""

  def __init__(self):

    #----------------------------------------------------------------------------
    # Notice that 'name_scope' and 'variable_scope' are different things.
    #
    # 'name_scope' is used to group several variables
    # (as long as you always use tf.get_variable() instead of tf.Variable()).
    # When you visualize your graph on TensorBoard, they are bundled together
    # and are shown as one box with rounded corners, called 'Namespace'.
    #
    # 'variable_scope' is used to share or unshare variables.
    # Two variables with the same name in the same variable_scope can
    # share their values if you set reuse=True, or raise "already exists" error.
    # If variable_scope differs, tf.get_variable() creates a distinct variable.
    # This allows us to reuse variable creation functions. You just change
    # the variable_scope and call the function to create different variable sets.
    #----------------------------------------------------------------------------
    with tf.name_scope('train_input'):
      self.input_ph = tf.placeholder(tf.int32, [None, T], name="input_ph")
      self.target_ph = tf.placeholder(tf.int32, [None, T], name="target_ph")

      embedding = tf.get_variable("embedding", [vocab_size, H], initializer=tf.random_normal_initializer(), dtype=tf.float32)

      # input_ph is B x T.
      # input_embedded is B x T x H.
      input_embedded = tf.nn.embedding_lookup(embedding, self.input_ph)

    self.cell = tf.contrib.rnn.GRUCell(H)

    # Return zero-filled state tensor of shape [self.initial_state_batch_size x state_size]
    self.initial_state_batch_size = tf.placeholder(tf.int32, shape=(), name="initial_state_batch_size")
    self.initial_state = self.cell.zero_state(self.initial_state_batch_size, dtype=tf.float32)
    
    self.state_ph = tf.placeholder(tf.float32, (None, self.cell.state_size), name="state_ph")

    # input_embedded : B x T x H
    # output: B x T x H
    # state : B x cell.state_size
    with tf.variable_scope("rnn"):
      output, state = tf.nn.dynamic_rnn(
          self.cell, input_embedded, initial_state=self.state_ph
          #,sequence_length=sequence_length
          )
    # Save last state. This will be the initial state for the next batch.
    self.final_state = state

    # reshape to (B * T) x H.
    output_flat = tf.reshape(output, [-1, H])

    # Convert hidden layer's output to the vector of logits for each vocabulary.
    softmax_w = tf.get_variable("softmax_w", [H, vocab_size], dtype=tf.float32)
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
    logits = tf.matmul(output_flat, softmax_w) + softmax_b

    #-----------------
    # inference
    #-----------------
    with tf.name_scope('infer'):
      self.infer_input_ph = tf.placeholder(tf.int32, [1, 1], name="infer_input_ph")
      infer_input_embedded = tf.nn.embedding_lookup(embedding, self.infer_input_ph)
      with tf.variable_scope("rnn", reuse=True):
        infer_output, infer_state = tf.nn.dynamic_rnn(
            self.cell,
            infer_input_embedded,
            initial_state=self.state_ph)
      self.final_infer_state = infer_state
      # infer_output_flat's shape is 1 x H
      infer_output_flat = tf.reshape(infer_output, [1, -1], name="infer_output_flat")
      # infer_logits's shape is 1 x vocab_size
      infer_logits = tf.matmul(infer_output_flat, softmax_w) + softmax_b
      # logit of the next char given he preceding chars.
      self.next_idx_prob = tf.nn.softmax(infer_logits[0])
    
    with tf.name_scope('loss'):
      # cross_entropy is a vector of length B * T
      cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(self.target_ph, [-1]), logits=logits)
      self.loss = tf.reduce_mean(cross_entropy)
    #------------------------------------------------
    # Step 1: in your TensorFlow graph, annotate
    # nodes with summary operations.
    #------------------------------------------------
    cross_entropy_summary = tf.summary.scalar('cross_entropy', self.loss)

    with tf.name_scope('optimize'):
      # Create an optimizer.
      #optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
      optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

      # gradient clipping
      tvars = tf.trainable_variables()
      grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), max_grad_norm)
      self.training_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step())
  
    #------------------------------------------------
    # Step 2: merge summaries.
    # Combine all the summary nodes you've created
    # into a single op that generates all the summary
    # data.
    #------------------------------------------------
    self.merged_summary = tf.summary.merge_all()
    # You can also select the nodes to combine, like below.
    # self.merged_summary = tf.summary.merge([cross_entropy_summary])
    
    # We must initialize all variables before we use them.
    init = tf.global_variables_initializer()
    init.run()
  
  def get_initial_state(self, batch_size):
    # Return zero-filled state tensor of shape [batch_size x state_size]
    return sess.run(self.initial_state, feed_dict={self.initial_state_batch_size: batch_size})
      
  def train_batch(self, input_batch, target_batch, initial_state):
    #------------------------------------------------
    # Step 4: evaluate your merged summary node.
    #------------------------------------------------
    final_state_, _, final_loss, summary = sess.run([self.final_state, self.training_op, self.loss, self.merged_summary],
      feed_dict={
        self.input_ph: input_batch,
        self.target_ph: target_batch,
        self.state_ph: initial_state})
    return final_state_, final_loss, summary
      
  def get_next_idx_prob(self, seed_idx, state):
    return sess.run([self.next_idx_prob, self.final_infer_state],
      feed_dict={
        self.infer_input_ph: [[seed_idx]],
        self.state_ph: state})

  def predict(self, seed_idx, initial_state, len):
    str = ""
    state = initial_state
    for i in range(len):
      next_idx_prob_, state = self.get_next_idx_prob(seed_idx, state)
      seed_idx = np.random.choice(range(vocab_size), p=next_idx_prob_)
      str += ix_to_char[seed_idx]
    return str, state

sess = tf.InteractiveSession()
    
def main(_):
  print("log_dir is {0}".format(FLAGS.log_dir))
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)

  batch_stride = num_batch // B
  batch_head = 0
  
  model = Model()
  #-----------------------------------------------------
  # Step 3: create a summary writer.
  # This also adds the info about your TensorFlow graph
  # structure to the summary file.
  # You can visualize it on 'GRAPHS' tab afterwards.
  #-----------------------------------------------------
  summary_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)

  train_state = model.get_initial_state(B)
  predict_state = model.get_initial_state(1)
  seed_idx = 0
  
  for epoch in range(FLAGS.max_epoch):
  
    # prepare batch data
    idx = [(batch_head + x * batch_stride) % num_batch for x in range(0, B)]
    input_batch = [input_all[i] for i in idx]
    target_batch = [target_all[i] for i in idx]
    batch_head += 1
  
    train_state, last_loss, summary = model.train_batch(input_batch, target_batch, train_state)
    #---------------------------------------------------
    # Step 5: write the evaluated summary into the file.
    #---------------------------------------------------
    summary_writer.add_summary(summary, epoch)
    
    if epoch % 50 == 0:
      print('epoch {0}: loss = {1:.3f} (perplexity = {2})'.format(epoch, last_loss, math.exp(last_loss)))

    if epoch % 100 == 0:
      str, predict_state = model.predict(seed_idx, predict_state, 200)
      print("")
      print(str)
      print("")
      seed_idx = char_to_ix[str[-1]]
  #-----------------------------------
  # Step 6: close the summary writer.
  #-----------------------------------
  summary_writer.close()
  
if __name__ == '__main__':
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

  