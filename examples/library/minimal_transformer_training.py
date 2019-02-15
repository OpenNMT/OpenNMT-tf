"""This example demonstrates how to train a standard Transformer model using
OpenNMT-tf as a library in about 100 lines of code. It mostly showcases the API
usage and is not intended to be used for actual trainings.
"""

import tensorflow as tf
import opennmt as onmt

from opennmt.utils.losses import cross_entropy_sequence_loss
from opennmt.utils.decay import noam_decay_v2


# Training files.
source_vocab = "/home/klein/data/wmt-ende/wmtende.vocab"
target_vocab = "/home/klein/data/wmt-ende/wmtende.vocab"
source_file = "/home/klein/data/wmt-ende/train.en"
target_file = "/home/klein/data/wmt-ende/train.de"

# Create the source and target inputters.
source_inputter = onmt.inputters.WordEmbedder("source_vocabulary", embedding_size=512)
target_inputter = onmt.inputters.WordEmbedder("target_vocabulary", embedding_size=512)

# Create the training inputter.
inputter = onmt.inputters.ExampleInputter(source_inputter, target_inputter)

# Initialize the vocabularies.
inputter.initialize({
    "source_vocabulary": source_vocab,
    "target_vocabulary": target_vocab
})

# Create the training dataset.
dataset = inputter.make_training_dataset(
    source_file,
    target_file,
    batch_size=3072,
    batch_type="tokens",
    shuffle_buffer_size=1000000,  # Shuffle 1M consecutive examples.
    bucket_width=1,  # Bucketize sequences by the same length for efficiency.
    maximum_features_length=100,
    maximum_labels_length=100)

# Create the dataset iterator.
iterator = dataset.make_initializable_iterator()
source, target = iterator.get_next()

# Create the encoder and decoder objects.
encoder = onmt.encoders.SelfAttentionEncoder(num_layers=6)
decoder = onmt.decoders.SelfAttentionDecoder(num_layers=6)
mode = tf.estimator.ModeKeys.TRAIN

# Encode the source.
with tf.variable_scope("encoder"):
  source_embedding = source_inputter.make_inputs(source, training=True)
  memory, _, _ = encoder.encode(source_embedding, source["length"], mode=mode)

# Decode the target.
with tf.variable_scope("decoder"):
  target_embedding = target_inputter.make_inputs(target, training=True)
  logits, _, _ = decoder.decode(
      target_embedding,
      target["length"],
      vocab_size=target_inputter.vocabulary_size,
      mode=mode,
      memory=memory,
      memory_sequence_length=source["length"])

# Compute the loss.
loss, normalizer, _ = cross_entropy_sequence_loss(
    logits,
    target["ids_out"],
    target["length"],
    label_smoothing=0.1,
    average_in_time=True,
    mode=mode)
loss /= normalizer

# Define the learning rate schedule.
step = tf.train.create_global_step()
learning_rate = tf.constant(2.0, dtype=tf.float32)
learning_rate = noam_decay_v2(learning_rate, step, model_dim=512, warmup_steps=4000)

# Define the optimization op.
optimizer = tf.contrib.opt.LazyAdamOptimizer(learning_rate)
train_op = optimizer.minimize(loss, global_step=step)

# Runs the training loop.
saver = tf.train.Saver()
report_every = 50
save_every = 5000
train_steps = 100000
ckpt_prefix = "transformer/model"

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  sess.run(tf.tables_initializer())
  sess.run(iterator.initializer)
  while True:
    step_, lr_, loss_, _ = sess.run([step, learning_rate, loss, train_op])
    if step_ % report_every == 0:
      print("Step = %d ; Learning rate = %f ; Loss = %f" % (step_, lr_, loss_))
    if step_ % save_every == 0:
      print("Saving checkpoint for step %d" % step_)
      saver.save(sess, ckpt_prefix, global_step=step_)
    if step_ == train_steps:
      break
