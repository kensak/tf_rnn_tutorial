"""
Simple character-level RNN model using TensorFlow.
Tutorial 1

Copyright (c) Tsuyoshi Matsumoto 2017.

Dual licensed under the MIT license http://www.opensource.org/licenses/mit-license.php
and GPL v2 license http://www.gnu.org/licenses/gpl.html .
 
"""
import tensorflow as tf
import numpy as np
import math

# hyperparameters
B = 20
hidden_size = 200 # size of hidden layer of neurons
T = 25            # number of time steps to unroll the RNN for
learning_rate = 0.001
num_training_epoch = 20000
max_grad_norm = 5 # for gradient clipping

#----------------
# prepare data
#----------------

# should be simple plain text file
data = open('data/KoizumiYagumo_all.txt', 'r').read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has {0} characters, {1} unique.'.format(data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

input_index_raw = np.array([char_to_ix[ch] for ch in data])
input_index_raw = input_index_raw[0:len(input_index_raw) // T * T]
input_index_raw_shift = np.append(input_index_raw[1:], input_index_raw[0])
input_all = input_index_raw.reshape([-1, T])
target_all = input_index_raw_shift.reshape([-1, T])
num_batch = len(input_all)

#----------------
# build model
#----------------

input_ph = tf.placeholder(tf.int32, [None, T], name="input_ph")
target_ph = tf.placeholder(tf.int32, [None, T], name="target_ph")

embedding = tf.get_variable("embedding", [vocab_size, hidden_size], initializer=tf.random_normal_initializer(), dtype=tf.float32)

# input_ph is B x num_steps.
# input_embedded is B x num_steps x hidden_size.
input_embedded = tf.nn.embedding_lookup(embedding, input_ph)

cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=True)

# Return zero-filled state tensor of shape [B x state_size]
initial_state = cell.zero_state(B, dtype=tf.float32)

cstate_ph = tf.placeholder(tf.float32, (None, cell.state_size[0]), name="cstate_ph")
mstate_ph = tf.placeholder(tf.float32, (None, cell.state_size[1]), name="mstate_ph")

# input_embedded : B x T x ...
# output: B x T x cell.output_size
# state : B x cell.state_size
output, state = tf.nn.dynamic_rnn(cell, input_embedded, initial_state=tf.contrib.rnn.LSTMStateTuple(cstate_ph, mstate_ph))
# Save last state. This will be the initial state for the next batch.
final_state = state

# reshape to (B * T) x hidden_size.
output_flat = tf.reshape(output, [-1, hidden_size])

# Convert hidden layer's output to logits for each vocabulary.
softmax_w = tf.get_variable("softmax_w", [hidden_size, vocab_size], dtype=tf.float32)
softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
logits = tf.matmul(output_flat, softmax_w) + softmax_b

# Used for inference.
# In inference, batch size is 1.
# logits is T x vocab_size matrix.
# We take only the last vector of length vocab_size,
# which is the logit of the next char given T preceding chars.
next_idx_prob = tf.nn.softmax(logits[-1])

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(target_ph, [-1]), logits=logits)
  
# cross_entropy is a vector of length B * T
loss = tf.reduce_mean(cross_entropy)

# Create an optimizer.
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

# gradient clipping
tvars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), max_grad_norm)
training_op = optimizer.apply_gradients(
  zip(grads, tvars),
  global_step=tf.contrib.framework.get_or_create_global_step())

#----------------
# train model
#----------------
  
init = tf.global_variables_initializer()

with tf.Session() as sess:
  # We must initialize all variables before we use them.
  init.run()

  batch_stride = num_batch // B
  batch_head = 0
  next_state = sess.run(initial_state)
  
  for epoch in range(num_training_epoch):
  
    # prepare batch data
    idx = [(batch_head + x * batch_stride) % num_batch for x in range(0, B)]
    input_batch = [input_all[i] for i in idx]
    target_batch = [target_all[i] for i in idx]
    batch_head += 1
  
    next_state, _, last_loss = sess.run([final_state, training_op, loss],
      feed_dict={input_ph: input_batch, target_ph: target_batch, cstate_ph: next_state[0], mstate_ph: next_state[1]})
    
    if epoch % 50 == 0:
      print('epoch {0}: loss = {1:.3f} (perplexity = {2})'.format(epoch, last_loss, math.exp(last_loss)))

    if epoch % 250 == 0:
      print("")
      input_batch = input_index_raw[0:T]
      for i in range(300):
        next_idx_prob_ = sess.run(next_idx_prob,
          feed_dict={input_ph: [input_batch], cstate_ph: next_state[0][0:1], mstate_ph: next_state[1][0:1]})

        # Select next char according to distribution next_idx_prob_.
        next_char_idx = np.random.choice(range(vocab_size), p=next_idx_prob_)
        
        print(ix_to_char[next_char_idx], end="")
        input_batch = input_batch[1:T]
        input_batch = np.append(input_batch, next_char_idx)
      print("\n")
