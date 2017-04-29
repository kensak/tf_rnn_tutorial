"""
Simple character-level RNN model using TensorFlow.
Tutorial 3 - Saving and Restoring Learned Parameters

Copyright (c) Tsuyoshi Matsumoto 2017.

Dual licensed under the MIT license http://www.opensource.org/licenses/mit-license.php
and GPL v2 license http://www.gnu.org/licenses/gpl.html .

TensorBoard allows us to visualize our TensorFlow graph, plot quantitative metrics about
the execution of your graph, and show additional data like images that pass through it.
https://www.tensorflow.org/get_started/summaries_and_tensorboard

------------------------------------------------------------------------------------

If you want to resume your training exactly at the point you stopped last time,
you need to save and restore the following data.
1. all the TensorFlow variables (learned parameters)
2. the states of the RNN
3. the position of the batch data in the whole input data

You can save the tf variables by
1-A. creating tf.train.Saver object, and
1-B. saving a checkpoint at some interval.

To save the states of RNN, make them tf variable.
The code below shows how.

To save and restore the batch position, define 'global_step' variable.
By passing this variable to the optimizer's 'minimize' function,
it will be incremented each time you optimize the parameters.
Use its evaluated value to calculate the position of a batch.
On restoring, 'global_step' variable will also be restored
and your batch pointer will also move to the right position
automatically.

"""
import argparse
import sys
import os
import tensorflow as tf
import numpy as np
import math

# hyperparameters
B = 20  # batch size
H = 200 # size of hidden layer of neurons
T = 25  # number of time steps to unroll the RNN for

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
                    help='Path to a plain text file')
parser.add_argument('--log_dir', type=str, default='log',
                    help='Summaries log directory')
parser.add_argument('--checkpoint_dir',
                    default='tmp',
                    help='Directory for storing checkpoint files')
parser.add_argument('--max_grad_norm', type=int, default=5,
                    help="Max value for gradient clipping. If zero, won't clip.")
                    
FLAGS, unparsed = parser.parse_known_args()

#----------------
# prepare data
#----------------

data = open(FLAGS.data_file, 'r').read()
#------------------------------------------------------------------------------
# CAUTION! You shouldn't write
#   chars = list(set(data))
# because every time you restart python, 'chars' would have different ordering,
# hence the vocabulary would have different indices.
# Python uses randomized hash function to build the set, which results in
# different ordering as we list-ify the set.
#------------------------------------------------------------------------------
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
  """RNN character base language model."""

  def __init__(self):

    with tf.name_scope('train_input'):
      self.input_ph = tf.placeholder(tf.int32, [None, T], name="input_ph")
      self.target_ph = tf.placeholder(tf.int32, [None, T], name="target_ph")

      embedding = tf.get_variable("embedding", [vocab_size, H], initializer=tf.random_normal_initializer(), dtype=tf.float32)

      # input_ph is B x T.
      # input_embedded is B x T x H.
      input_embedded = tf.nn.embedding_lookup(embedding, self.input_ph)

    cell = tf.contrib.rnn.GRUCell(H)
    self.infer_zero_state = cell.zero_state(1, dtype=tf.float32)
    
    self.state_ph = tf.placeholder(tf.float32, (None, cell.state_size), name="state_ph")
    
    # 2. Make state variable so that it will be saved by the saver.
    #
    # Evaluate this tf variable to get the initial state.
    # If there is no checkpoint to restore,
    # this evaluates to the zero-initialized state.
    # Otherwise, this evaluates to the restored state.
    self.state = tf.get_variable(
      "state",
      (B, cell.state_size),
      initializer=tf.zeros_initializer(),
      trainable=False,
      dtype=tf.float32)

    # input_embedded : B x T x H
    # output: B x T x H
    # state : B x cell.state_size
    with tf.variable_scope("rnn"):
      output, state_ = tf.nn.dynamic_rnn(
        cell,
        input_embedded,
        initial_state=self.state_ph
        )
    self.final_state = tf.assign(self.state, state_)

    # reshape to (B * T) x H.
    output_flat = tf.reshape(output, [-1, H])

    # Convert hidden layer's output to the vector of logits for each vocabulary.
    softmax_w = tf.get_variable("softmax_w", [H, vocab_size], dtype=tf.float32)
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
    logits = tf.matmul(output_flat, softmax_w) + softmax_b

    #-----------------
    # inference
    # Given the index of seed character in infer_input_ph, 
    # calculate the next char's probability distribution and  set it to next_idx_prob.
    #-----------------
    with tf.name_scope('infer'):
      self.infer_input_ph = tf.placeholder(tf.int32, [1, 1], name="infer_input_ph")
      infer_input_embedded = tf.nn.embedding_lookup(embedding, self.infer_input_ph)
      with tf.variable_scope("rnn", reuse=True):
        infer_output, infer_state = tf.nn.dynamic_rnn(
            cell,
            infer_input_embedded,
            initial_state=self.state_ph
            )
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
    cross_entropy_summary = tf.summary.scalar('cross_entropy', self.loss)

    with tf.name_scope('optimize'):
      # Create an optimizer.
      #optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
      optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

      # 3. Define global_step.
      self.global_step = tf.get_variable("global_step", (), initializer=tf.zeros_initializer(), trainable=False, dtype=tf.int32)
      if FLAGS.max_grad_norm == 0:
        self.training_op = optimizer.minimize(cross_entropy, global_step=self.global_step)
      else:
        # gradient clipping
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), FLAGS.max_grad_norm)
        self.training_op = optimizer.apply_gradients(
          zip(grads, tvars),
          global_step=self.global_step)
        
    # Merge summaries
    self.train_merged_summary = tf.summary.merge([cross_entropy_summary])
    #self.merged = tf.summary.merge_all()
  
  def train_batch(self, input_batch, target_batch, initial_state):
    final_state_, _, final_loss, summary = sess.run(
      [self.final_state, self.training_op, self.loss, self.train_merged_summary],
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
    #    self.isRestore: False})

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
  
  print("checkpoint_dir is {0}".format(FLAGS.checkpoint_dir))
  if not tf.gfile.Exists(FLAGS.checkpoint_dir):
    tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

  batch_stride = num_batch // B

  # Create the TansorFlow graph.
  model = Model()
  # 1-A. Create tf.train.Saver object.
  # This should come after the creation of the graph.
  saver = tf.train.Saver()

  ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
  if ckpt:
    last_model = ckpt.model_checkpoint_path
    print("load " + last_model)
    saver.restore(sess, last_model)
  else:
    init = tf.global_variables_initializer()
    init.run()
    if tf.gfile.Exists(FLAGS.log_dir):
      tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)

  # Add one in order not to repeat the last training step twice.
  epoch = tf.train.global_step(sess, model.global_step) + 1
    
  # If there is no checkpoint,
  # this is zero-initialized state.
  # Otherwise, this is the restored state.
  train_state = model.state.eval()
  predict_state = model.infer_zero_state.eval()
  
  train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
  
  seed_idx = 0
  
  while True:
    if epoch > FLAGS.max_epoch:
      break
  
    # prepare batch data
    idx = [(epoch + x * batch_stride) % num_batch for x in range(0, B)]
    input_batch = input_all[idx]
    target_batch = target_all[idx]
  
    train_state, last_loss, summary = model.train_batch(input_batch, target_batch, train_state)
    train_writer.add_summary(summary, epoch)
    
    if epoch % 50 == 0:
      print('epoch {0}: loss = {1:.3f} (perplexity = {2})'.format(epoch, last_loss, math.exp(last_loss)))

    if epoch % 250 == 0:
      str, predict_state = model.predict(seed_idx, predict_state, 200)
      print("")
      print(str)
      print("")
      seed_idx = char_to_ix[str[-1]]
      # 1-B. Save a checkpoint at some interval.
      saver.save(sess, os.path.join(FLAGS.checkpoint_dir, "model.ckpt"), global_step=epoch)
      
    epoch = epoch + 1
    
  train_writer.close()
  
  saver.save(sess, os.path.join(FLAGS.checkpoint_dir, "model.ckpt"), global_step=epoch)
  
if __name__ == '__main__':
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

  