# -*- coding: utf-8 -*-

import os

from parameterized import parameterized
from numbers import Number

import tensorflow as tf

from opennmt import decoders
from opennmt import encoders
from opennmt import estimator
from opennmt import inputters
from opennmt import models
from opennmt.models import catalog
from opennmt.tests import test_util


def _seq2seq_model(mode):
  model = models.SequenceToSequence(
      inputters.WordEmbedder(16),
      inputters.WordEmbedder(16),
      encoders.SelfAttentionEncoder(2, 16, 4, 32),
      decoders.SelfAttentionDecoder(2, 16, 4, 32))
  params = {}
  if mode == tf.estimator.ModeKeys.TRAIN:
    params["optimizer"] = "SGD"
    params["learning_rate"] = 0.1
  return model, params


class ModelTest(tf.test.TestCase):

  def _makeToyEnDeData(self, with_alignments=False):
    data_config = {}
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
    data_config["source_vocabulary"] = test_util.make_vocab_from_file(
        os.path.join(self.get_temp_dir(), "src_vocab.txt"), features_file)
    data_config["target_vocabulary"] = test_util.make_vocab_from_file(
        os.path.join(self.get_temp_dir(), "tgt_vocab.txt"), labels_file)
    if with_alignments:
      # Dummy and incomplete alignments.
      data_config["train_alignments"] = test_util.make_data_file(
          os.path.join(self.get_temp_dir(), "aligne.txt"),
          ["0-0 1-0 2-2 3-4 4-4 5-6",
           "0-1 1-1 1-3 2-3 4-4",
           "0-0 1-0 2-2 3-4 4-4 5-6"])
    return features_file, labels_file, data_config

  def _makeToyLMData(self):
    features_file, _, data_config = self._makeToyEnDeData()
    return features_file, {"vocabulary": data_config["source_vocabulary"]}

  def _makeToyTaggerData(self):
    data_config = {}
    features_file = test_util.make_data_file(
        os.path.join(self.get_temp_dir(), "src.txt"),
        ["M . Smith went to Washington .",
         "I live in New Zealand ."])
    labels_file = test_util.make_data_file(
        os.path.join(self.get_temp_dir(), "labels.txt"),
        ["B-PER I-PER E-PER O O S-LOC O",
         "O O O B-LOC E-LOC O"])
    data_config["source_vocabulary"] = test_util.make_vocab_from_file(
        os.path.join(self.get_temp_dir(), "src_vocab.txt"), features_file)
    data_config["target_vocabulary"] = test_util.make_data_file(
        os.path.join(self.get_temp_dir(), "labels_vocab.txt"),
        ["O", "B-LOC", "I-LOC", "E-LOC", "S-LOC", "B-PER", "I-PER", "E-PER", "S-PER"])
    return features_file, labels_file, data_config

  def _makeToyClassifierData(self):
    data_config = {}
    features_file = test_util.make_data_file(
        os.path.join(self.get_temp_dir(), "src.txt"),
        ["This product was not good at all , it broke on the first use !",
         "Perfect , it does everything I need .",
         "How do I change the battery ?"])
    labels_file = test_util.make_data_file(
        os.path.join(self.get_temp_dir(), "labels.txt"), ["negative", "positive", "neutral"])
    data_config["source_vocabulary"] = test_util.make_vocab_from_file(
        os.path.join(self.get_temp_dir(), "src_vocab.txt"), features_file)
    data_config["target_vocabulary"] = test_util.make_data_file(
        os.path.join(self.get_temp_dir(), "labels_vocab.txt"), ["negative", "positive", "neutral"])
    return features_file, labels_file, data_config

  def _testGenericModel(self,
                        model,
                        mode,
                        features_file,
                        labels_file=None,
                        data_config=None,
                        batch_size=16,
                        prediction_heads=None,
                        metrics=None,
                        params=None):
    # Mainly test that the code does not throw.
    if params is None:
      params = model.auto_config()["params"]
    if data_config is None:
      data_config = {}
    model.initialize(data_config)
    with tf.Graph().as_default():
      dataset = estimator.make_input_fn(
          model,
          mode,
          batch_size,
          features_file,
          labels_file=labels_file if mode != tf.estimator.ModeKeys.PREDICT else None)()
      iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
      data = iterator.get_next()
      if mode != tf.estimator.ModeKeys.PREDICT:
        features, labels = data
      else:
        features, labels = data, None
      estimator_spec = estimator.make_model_fn(model)(features, labels, params, mode, None)
      with self.test_session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())
        sess.run(tf.compat.v1.tables_initializer())
        sess.run(iterator.initializer)
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
    model, params = _seq2seq_model(mode)
    features_file, labels_file, data_config = self._makeToyEnDeData()
    self._testGenericModel(
        model,
        mode,
        features_file,
        labels_file,
        data_config,
        prediction_heads=["tokens", "length", "log_probs"],
        params=params)

  @parameterized.expand([["ce"], ["mse"]])
  def testSequenceToSequenceWithGuidedAlignment(self, ga_type):
    mode = tf.estimator.ModeKeys.TRAIN
    model, params = _seq2seq_model(mode)
    params["guided_alignment_type"] = ga_type
    features_file, labels_file, data_config = self._makeToyEnDeData(with_alignments=True)
    model.initialize(data_config)
    with tf.Graph().as_default():
      dataset = estimator.make_input_fn(model, mode, 16, features_file, labels_file)()
      iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
      features, labels = iterator.get_next()
      self.assertIn("alignment", labels)
      estimator_spec = estimator.make_model_fn(model)(features, labels, params, mode, None)
      with self.session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())
        sess.run(tf.compat.v1.tables_initializer())
        sess.run(iterator.initializer)
        loss = sess.run(estimator_spec.loss)
        self.assertIsInstance(loss, Number)

  def testSequenceToSequenceWithReplaceUnknownTarget(self):
    mode = tf.estimator.ModeKeys.PREDICT
    model, params = _seq2seq_model(mode)
    params["replace_unknown_target"] = True
    features_file, labels_file, data_config = self._makeToyEnDeData()
    model.initialize(data_config)
    with tf.Graph().as_default():
      dataset = estimator.make_input_fn(model, mode, 16, features_file, labels_file)()
      iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
      features = iterator.get_next()
      estimator_spec = estimator.make_model_fn(model)(features, None, params, mode, None)
      with self.session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())
        sess.run(tf.compat.v1.tables_initializer())
        sess.run(iterator.initializer)
        _ = sess.run(estimator_spec.predictions)

  def testSequenceToSequenceServing(self):
    # Test that serving features can be forwarded into the model.
    mode = tf.estimator.ModeKeys.PREDICT
    _, _, data_config = self._makeToyEnDeData()
    model, params = _seq2seq_model(mode)
    model.initialize(data_config)
    with tf.Graph().as_default():
      features = estimator.make_serving_input_fn(model)().features
      _, predictions = model(features, None, params, mode)
      self.assertIsInstance(predictions, dict)

  @parameterized.expand([
      [tf.estimator.ModeKeys.TRAIN],
      [tf.estimator.ModeKeys.EVAL],
      [tf.estimator.ModeKeys.PREDICT]])
  def testLanguageModel(self, mode):
    # Mainly test that the code does not throw.
    decoder = decoders.SelfAttentionDecoder(
        2, num_units=16, num_heads=4, ffn_inner_dim=32, num_sources=0)
    model = models.LanguageModel(decoder, embedding_size=16)
    features_file, data_config = self._makeToyLMData()
    params = {
        "optimizer": "SGD",
        "learning_rate": 0.1}
    self._testGenericModel(
        model,
        mode,
        features_file,
        data_config=data_config,
        batch_size=1 if mode == tf.estimator.ModeKeys.PREDICT else 16,
        prediction_heads=["tokens", "length"],
        params=params)

  @parameterized.expand([
      [tf.estimator.ModeKeys.TRAIN],
      [tf.estimator.ModeKeys.EVAL],
      [tf.estimator.ModeKeys.PREDICT]])
  def testSequenceTagger(self, mode):
    model = models.SequenceTagger(inputters.WordEmbedder(10), encoders.MeanEncoder())
    features_file, labels_file, data_config = self._makeToyTaggerData()
    data_config["tagging_scheme"] = "bioes"
    params = {
        "optimizer": "SGD",
        "learning_rate": 0.1}
    self._testGenericModel(
        model,
        mode,
        features_file,
        labels_file,
        data_config,
        prediction_heads=["tags", "length"],
        metrics=["accuracy", "precision", "recall", "f1"],
        params=params)

  @parameterized.expand([
      [tf.estimator.ModeKeys.TRAIN],
      [tf.estimator.ModeKeys.EVAL],
      [tf.estimator.ModeKeys.PREDICT]])
  def testSequenceClassifier(self, mode):
    model = models.SequenceClassifier(inputters.WordEmbedder(10), encoders.MeanEncoder())
    features_file, labels_file, data_config = self._makeToyClassifierData()
    params = {
        "optimizer": "SGD",
        "learning_rate": 0.1}
    self._testGenericModel(
        model,
        mode,
        features_file,
        labels_file,
        data_config,
        prediction_heads=["classes"],
        metrics=["accuracy"],
        params=params)

if __name__ == "__main__":
  tf.test.main()
