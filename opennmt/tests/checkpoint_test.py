import os
import six
import time

import tensorflow as tf
import numpy as np

from opennmt.utils import checkpoint
from opennmt.utils.vocab import Vocab
from opennmt.tests import test_util


class CheckpointTest(tf.test.TestCase):

  def _saveVocab(self, name, words):
    vocab = Vocab()
    for word in words:
      vocab.add(str(word))
    vocab_file = os.path.join(self.get_temp_dir(), name)
    vocab.serialize(vocab_file)
    return vocab_file

  def testVocabMappingMerge(self):
    old = self._saveVocab("old", ["1", "2", "3", "4"])
    new = self._saveVocab("new", ["1", "6", "3", "5", "7"])
    mapping, new_vocab = checkpoint._get_vocabulary_mapping(old, new, "merge")
    self.assertEqual(4 + 5 - 2 + 1, len(mapping))  # old + new - common + <unk>
    self.assertAllEqual([0, 1, 2, 3, -1, -1, -1, 4], mapping)
    self.assertAllEqual(["1", "2", "3", "4", "6", "5", "7"], new_vocab.words)

  def testVocabMappingReplace(self):
    old = self._saveVocab("old", ["1", "2", "3", "4"])
    new = self._saveVocab("new", ["1", "6", "5", "3", "7"])
    mapping, new_vocab = checkpoint._get_vocabulary_mapping(old, new, "replace")
    self.assertEqual(5 + 1, len(mapping))  # new + <unk>
    self.assertAllEqual([0, -1, -1, 2, -1, 4], mapping)
    self.assertAllEqual(["1", "6", "5", "3", "7"], new_vocab.words)

  def testVocabVariableUpdate(self):
    mapping = [0, -1, -1, 2, -1, 4]
    old = np.array([1, 2, 3, 4, 5, 6, 7])
    vocab_size = 7
    new = checkpoint._update_vocabulary_variable(old, vocab_size, mapping)
    self.assertAllEqual([1, 0, 0, 3, 0, 5], new)

  def _generateCheckpoint(self,
                          model_dir,
                          step,
                          variables,
                          last_checkpoints=None,
                          prefix="model.ckpt"):
    with tf.Graph().as_default() as graph:
      for name, value in six.iteritems(variables):
        if isinstance(value, tuple):
          dtype = None
          initializer = tf.random_uniform_initializer()
          shape = value
        else:
          dtype = tf.as_dtype(value.dtype)
          initializer = tf.constant_initializer(value, dtype=dtype)
          shape = value.shape
        _ = tf.get_variable(
            name,
            shape=shape,
            dtype=dtype,
            initializer=initializer)
      global_step = tf.get_variable(
          "global_step",
          initializer=tf.constant(step, dtype=tf.int64),
          trainable=False)
      saver = tf.train.Saver(tf.global_variables())
      if last_checkpoints:
        saver.set_last_checkpoints_with_time(last_checkpoints)
      with self.test_session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        saver.save(sess, os.path.join(model_dir, prefix), global_step=global_step)
      return saver.last_checkpoints[0], time.time()

  @test_util.run_tf1_only
  def testCheckpointAveraging(self):
    model_dir = os.path.join(self.get_temp_dir(), "ckpt")
    os.makedirs(model_dir)
    checkpoints = []
    checkpoints.append(self._generateCheckpoint(
        model_dir, 10,
        {"x": np.zeros((2, 3), dtype=np.float32), "words_per_sec/features_init": np.int64(42)},
        last_checkpoints=checkpoints))
    checkpoints.append(self._generateCheckpoint(
        model_dir, 20,
        {"x": np.ones((2, 3), dtype=np.float32), "words_per_sec/features_init": np.int64(89)},
        last_checkpoints=checkpoints))
    avg_dir = os.path.join(model_dir, "avg")
    checkpoint.average_checkpoints(model_dir, avg_dir)
    avg_var = checkpoint.get_checkpoint_variables(avg_dir)
    self.assertEqual(avg_var["global_step"].dtype, np.int64)
    self.assertEqual(avg_var["global_step"], 20)
    self.assertEqual(avg_var["words_per_sec/features_init"].dtype, np.int64)
    self.assertEqual(avg_var["words_per_sec/features_init"], 89)
    self.assertAllEqual(avg_var["x"], np.full((2, 3), 0.5, dtype=np.float32))

  @test_util.run_tf1_only
  def testCheckpointDTypeConversion(self):
    model_dir = os.path.join(self.get_temp_dir(), "ckpt-fp32")
    os.makedirs(model_dir)
    variables = {
      "x": np.ones((2, 3), dtype=np.float32),
      "optim/x": np.ones((2, 3), dtype=np.float32),
      "counter": np.int64(42)
    }
    checkpoint_path, _ = self._generateCheckpoint(model_dir, 10, variables)
    half_dir = os.path.join(model_dir, "fp16")
    checkpoint.convert_checkpoint(checkpoint_path, half_dir, tf.float32, tf.float16)
    half_var = checkpoint.get_checkpoint_variables(half_dir)
    self.assertEqual(half_var["global_step"], 10)
    self.assertEqual(half_var["x"].dtype, np.float16)
    self.assertEqual(half_var["optim/x"].dtype, np.float32)
    self.assertEqual(half_var["counter"].dtype, np.int64)


if __name__ == "__main__":
  tf.test.main()
