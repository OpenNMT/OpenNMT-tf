# -*- coding: utf-8 -*-

import os

from parameterized import parameterized
from numbers import Number

import tensorflow as tf

from opennmt import models, inputters, encoders
from opennmt.models import catalog
from opennmt.tests import test_util


@test_util.run_tf1_only
class ModelTest(tf.test.TestCase):

  def _makeToyEnDeData(self):
    metadata = {}
    features_file = test_util.make_data_file(
        os.path.join(self.get_temp_dir(), "src.txt"),
        ["Parliament Does Not Support Amendment Freeing Tymoshenko",
         "Today , the Ukraine parliament dismissed , within the Code of Criminal Procedure "
         "amendment , the motion to revoke an article based on which the opposition leader , "
         "Yulia Tymoshenko , was sentenced .",
         "The amendment that would lead to freeing the imprisoned former Prime Minister was "
         "revoked during second reading of the proposal for mitigation of sentences for "
         "economic offences ."])
    labels_file = test_util.make_data_file(
        os.path.join(self.get_temp_dir(), "tgt.txt"),
        ["Keine befreiende Novelle für Tymoshenko durch das Parlament",
         "Das ukrainische Parlament verweigerte heute den Antrag , im Rahmen einer Novelle "
         "des Strafgesetzbuches denjenigen Paragrafen abzuschaffen , auf dessen Grundlage die "
         "Oppositionsführerin Yulia Timoshenko verurteilt worden war .",
         "Die Neuregelung , die den Weg zur Befreiung der inhaftierten Expremierministerin hätte "
         "ebnen können , lehnten die Abgeordneten bei der zweiten Lesung des Antrags auf Milderung "
         "der Strafen für wirtschaftliche Delikte ab ."])
    metadata["source_words_vocabulary"] = test_util.make_vocab_from_file(
        os.path.join(self.get_temp_dir(), "src_vocab.txt"), features_file)
    metadata["target_words_vocabulary"] = test_util.make_vocab_from_file(
        os.path.join(self.get_temp_dir(), "tgt_vocab.txt"), labels_file)
    return features_file, labels_file, metadata

  def _makeToyTaggerData(self):
    metadata = {}
    features_file = test_util.make_data_file(
        os.path.join(self.get_temp_dir(), "src.txt"),
        ["M . Smith went to Washington .",
         "I live in New Zealand ."])
    labels_file = test_util.make_data_file(
        os.path.join(self.get_temp_dir(), "labels.txt"),
        ["B-PER I-PER E-PER O O S-LOC O",
         "O O O B-LOC E-LOC O"])
    metadata["source_vocabulary"] = test_util.make_vocab_from_file(
        os.path.join(self.get_temp_dir(), "src_vocab.txt"), features_file)
    metadata["target_vocabulary"] = test_util.make_data_file(
        os.path.join(self.get_temp_dir(), "labels_vocab.txt"),
        ["O", "B-LOC", "I-LOC", "E-LOC", "S-LOC", "B-PER", "I-PER", "E-PER", "S-PER"])
    return features_file, labels_file, metadata

  def _testGenericModel(self,
                        model,
                        mode,
                        features_file,
                        labels_file,
                        metadata,
                        batch_size=16,
                        prediction_heads=None,
                        metrics=None,
                        params=None):
    # Mainly test that the code does not throw.
    if params is None:
      params = model.auto_config()["params"]
    data = model.input_fn(
        mode,
        batch_size,
        metadata,
        features_file,
        labels_file=labels_file if mode != tf.estimator.ModeKeys.PREDICT else None)()
    if mode != tf.estimator.ModeKeys.PREDICT:
      features, labels = data
    else:
      features, labels = data, None
    estimator_spec = model.model_fn()(features, labels, params, mode, None)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      sess.run(tf.tables_initializer())
      if mode == tf.estimator.ModeKeys.TRAIN:
        loss = sess.run(estimator_spec.loss)
        self.assertIsInstance(loss, Number)
      elif mode == tf.estimator.ModeKeys.EVAL:
        fetches = [estimator_spec.loss]
        if estimator_spec.eval_metric_ops is not None:
          fetches.append(estimator_spec.eval_metric_ops)
        result = sess.run(fetches)
        self.assertIsInstance(result[0], Number)
        if metrics is not None:
          for metric in metrics:
            self.assertIn(metric, result[1])
      else:
        predictions = sess.run(estimator_spec.predictions)
        self.assertIsInstance(predictions, dict)
        if prediction_heads is not None:
          for head in prediction_heads:
            self.assertIn(head, predictions)

  @parameterized.expand([
      [tf.estimator.ModeKeys.TRAIN],
      [tf.estimator.ModeKeys.EVAL],
      [tf.estimator.ModeKeys.PREDICT]])
  def testSequenceToSequence(self, mode):
    # Mainly test that the code does not throw.
    model = catalog.NMTSmall()
    features_file, labels_file, metadata = self._makeToyEnDeData()
    self._testGenericModel(
        model,
        mode,
        features_file,
        labels_file,
        metadata,
        prediction_heads=["tokens", "length", "log_probs"])

  def testSequenceToSequenceServing(self):
    # Test that serving features can be forwarded into the model.
    model = catalog.NMTSmall()
    _, _, metadata = self._makeToyEnDeData()
    features = model.serving_input_fn(metadata)().features
    with tf.variable_scope(model.name):
      _, predictions = model(
          features, None, model.auto_config()["params"], tf.estimator.ModeKeys.PREDICT)
      self.assertIsInstance(predictions, dict)

  @parameterized.expand([
      [tf.estimator.ModeKeys.TRAIN],
      [tf.estimator.ModeKeys.EVAL],
      [tf.estimator.ModeKeys.PREDICT]])
  def testSequenceTagger(self, mode):
    model = models.SequenceTagger(
        inputters.WordEmbedder("source_vocabulary", 10),
        encoders.MeanEncoder(),
        "target_vocabulary",
        crf_decoding=True,
        tagging_scheme="bioes")
    features_file, labels_file, metadata = self._makeToyTaggerData()
    params = {
        "optimizer": "GradientDescentOptimizer",
        "learning_rate": 0.1}
    self._testGenericModel(
        model,
        mode,
        features_file,
        labels_file,
        metadata,
        prediction_heads=["tags", "length"],
        metrics=["accuracy", "precision", "recall", "f1"],
        params=params)


if __name__ == "__main__":
  tf.test.main()
