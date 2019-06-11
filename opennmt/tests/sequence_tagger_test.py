import tensorflow as tf
import numpy as np

from opennmt.models import sequence_tagger


class SequenceTaggerTest(tf.test.TestCase):

  def _testTagSchemeFlags(self,
                          tag_fn,
                          labels,
                          predicted,
                          expected_true_positives,
                          expected_false_positives,
                          expected_false_negatives):
    labels = np.array([[tf.compat.as_bytes(c) for c in labels]])
    predicted = np.array([[tf.compat.as_bytes(c) for c in predicted]])
    gold_flags, predicted_flags = tag_fn(labels, predicted)

    true_positives = tf.keras.metrics.TruePositives()
    false_positives = tf.keras.metrics.FalsePositives()
    false_negatives = tf.keras.metrics.FalseNegatives()
    true_positives.update_state(gold_flags, predicted_flags)
    false_positives.update_state(gold_flags, predicted_flags)
    false_negatives.update_state(gold_flags, predicted_flags)

    tp = self.evaluate(true_positives.result())
    fp = self.evaluate(false_positives.result())
    fn = self.evaluate(false_negatives.result())

    self.assertEqual(expected_true_positives, tp, msg="true positives mismatch")
    self.assertEqual(expected_false_positives, fp, msg="false positives mismatch")
    self.assertEqual(expected_false_negatives, fn, msg="false negatives mismatch")

  def testBIOESFlags(self):
    self._testTagSchemeFlags(
      sequence_tagger.flag_bioes_tags,
      ["S-LOC"], ["S-ORG"],
      expected_true_positives=0,
      expected_false_positives=1,
      expected_false_negatives=1)
    self._testTagSchemeFlags(
      sequence_tagger.flag_bioes_tags,
      ["B-LOC", "I-LOC", "E-LOC"], ["B-LOC", "I-LOC", "E-LOC"],
      expected_true_positives=1,
      expected_false_positives=0,
      expected_false_negatives=0)
    self._testTagSchemeFlags(
      sequence_tagger.flag_bioes_tags,
      ["O", "B-LOC", "I-LOC", "E-LOC"], ["B-LOC", "I-LOC", "E-LOC", "O"],
      expected_true_positives=0,
      expected_false_positives=1,
      expected_false_negatives=1)
    self._testTagSchemeFlags(
      sequence_tagger.flag_bioes_tags,
      ["B-LOC", "I-LOC", "E-LOC"], ["B-LOC", "E-LOC", "S-LOC"],
      expected_true_positives=0,
      expected_false_positives=2,
      expected_false_negatives=1)
    self._testTagSchemeFlags(
      sequence_tagger.flag_bioes_tags,
      ["B-LOC", "I-LOC", "E-LOC"], ["S-LOC", "O", "O"],
      expected_true_positives=0,
      expected_false_positives=1,
      expected_false_negatives=1)
    self._testTagSchemeFlags(
      sequence_tagger.flag_bioes_tags,
      ["S-LOC", "O"], ["B-LOC", "E-LOC"],
      expected_true_positives=0,
      expected_false_positives=1,
      expected_false_negatives=1)
    self._testTagSchemeFlags(
      sequence_tagger.flag_bioes_tags,
      ["B-ORG", "E-ORG",     "O", "B-PER", "E-PER", "O", "O", "O", "O", "B-MISC", "E-MISC",      "O"],
      ["B-ORG", "E-ORG", "S-PER", "S-PER",     "O", "O", "O", "O", "O",      "O",      "O", "S-MISC"],
      expected_true_positives=1,
      expected_false_positives=3,
      expected_false_negatives=2)


if __name__ == "__main__":
  tf.test.main()
