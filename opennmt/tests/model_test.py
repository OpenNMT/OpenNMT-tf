# -*- coding: utf-8 -*-

import os

from numbers import Number

import tensorflow as tf

from opennmt import constants
from opennmt.models import catalog
from opennmt.utils.vocab import Vocab
from opennmt.tests import test_util


def _make_vocab_from_file(path, data_file):
  vocab = Vocab(special_tokens=[
      constants.PADDING_TOKEN,
      constants.START_OF_SENTENCE_TOKEN,
      constants.END_OF_SENTENCE_TOKEN])
  vocab.add_from_text(data_file)
  vocab.serialize(path)
  return path

def _make_data_file(path, lines):
  with open(path, "w") as data:
    for line in lines:
      data.write("%s\n" % line)
  return path


@test_util.run_tf1_only
class ModelTest(tf.test.TestCase):

  def _makeToyEnDeData(self):
    metadata = {}
    features_file = _make_data_file(
        os.path.join(self.get_temp_dir(), "src.txt"),
        ["Parliament Does Not Support Amendment Freeing Tymoshenko",
         "Today , the Ukraine parliament dismissed , within the Code of Criminal Procedure "
         "amendment , the motion to revoke an article based on which the opposition leader , "
         "Yulia Tymoshenko , was sentenced .",
         "The amendment that would lead to freeing the imprisoned former Prime Minister was "
         "revoked during second reading of the proposal for mitigation of sentences for "
         "economic offences ."])
    labels_file = _make_data_file(
        os.path.join(self.get_temp_dir(), "tgt.txt"),
        ["Keine befreiende Novelle für Tymoshenko durch das Parlament",
         "Das ukrainische Parlament verweigerte heute den Antrag , im Rahmen einer Novelle "
         "des Strafgesetzbuches denjenigen Paragrafen abzuschaffen , auf dessen Grundlage die "
         "Oppositionsführerin Yulia Timoshenko verurteilt worden war .",
         "Die Neuregelung , die den Weg zur Befreiung der inhaftierten Expremierministerin hätte "
         "ebnen können , lehnten die Abgeordneten bei der zweiten Lesung des Antrags auf Milderung "
         "der Strafen für wirtschaftliche Delikte ab ."])
    metadata["source_words_vocabulary"] = _make_vocab_from_file(
        os.path.join(self.get_temp_dir(), "src_vocab.txt"), features_file)
    metadata["target_words_vocabulary"] = _make_vocab_from_file(
        os.path.join(self.get_temp_dir(), "tgt_vocab.txt"), labels_file)
    return features_file, labels_file, metadata

  def _testSequenceToSequence(self, mode):
    # Mainly test that the code does not throw.
    model = catalog.NMTSmall()
    features_file, labels_file, metadata = self._makeToyEnDeData()
    data = model.input_fn(
        mode,
        16,
        metadata,
        features_file,
        labels_file=labels_file if mode != tf.estimator.ModeKeys.PREDICT else None)()
    if mode != tf.estimator.ModeKeys.PREDICT:
      features, labels = data
    else:
      features, labels = data, None
    estimator_spec = model.model_fn()(
        features, labels, model.auto_config()["params"], mode, None)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())
      if mode != tf.estimator.ModeKeys.PREDICT:
        loss = sess.run(estimator_spec.loss)
        self.assertIsInstance(loss, Number)
      else:
        predictions = sess.run(estimator_spec.predictions)
        self.assertIsInstance(predictions, dict)
        self.assertIn("tokens", predictions)
        self.assertIn("length", predictions)
        self.assertIn("log_probs", predictions)

  def testSequenceToSequenceTraining(self):
    self._testSequenceToSequence(tf.estimator.ModeKeys.TRAIN)
  def testSequenceToSequenceEvaluation(self):
    self._testSequenceToSequence(tf.estimator.ModeKeys.EVAL)
  def testSequenceToSequenceInference(self):
    self._testSequenceToSequence(tf.estimator.ModeKeys.PREDICT)

  def testSequenceToSequenceServing(self):
    # Test that serving features can be forwarded into the model.
    model = catalog.NMTSmall()
    _, _, metadata = self._makeToyEnDeData()
    features = model.serving_input_fn(metadata)().features
    with tf.variable_scope(model.name):
      _, predictions = model(
          features, None, model.auto_config()["params"], tf.estimator.ModeKeys.PREDICT)
      self.assertIsInstance(predictions, dict)


if __name__ == "__main__":
  tf.test.main()
