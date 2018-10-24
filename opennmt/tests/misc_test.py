import tensorflow as tf
import numpy as np

from opennmt.utils import misc


class MiscTest(tf.test.TestCase):

  def testFormatTranslationOutput(self):
    self.assertEqual(
        misc.format_translation_output("hello world"),
        "hello world")
    self.assertEqual(
        misc.format_translation_output("hello world", score=42),
        "%f ||| hello world" % 42)
    self.assertEqual(
        misc.format_translation_output("hello world", score=42, token_level_scores=[24, 64]),
        "%f ||| hello world ||| %f %f" % (42, 24, 64))
    self.assertEqual(
        misc.format_translation_output("hello world", token_level_scores=[24, 64]),
        "hello world ||| %f %f" % (24, 64))
    self.assertEqual(
        misc.format_translation_output("hello world", attention=[[0.1, 0.7, 0.2], [0.5, 0.3, 0.2]]),
        "hello world")
    self.assertEqual(
        misc.format_translation_output(
            "hello world",
            attention=np.array([[0.1, 0.7, 0.2], [0.5, 0.3, 0.2]]),
            alignment_type="hard"),
        "hello world ||| 1-0 0-1")


if __name__ == "__main__":
  tf.test.main()
