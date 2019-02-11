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

  def testEventOrderRestorer(self):
    events = []
    restorer = misc.OrderRestorer(
        index_fn=lambda x: x[0],
        callback_fn=lambda x: events.append(x))
    restorer.push((2, "toto"))
    restorer.push((1, "tata"))
    restorer.push((3, "foo"))
    restorer.push((0, "bar"))
    restorer.push((4, "titi"))
    with self.assertRaises(ValueError):
      restorer.push((2, "invalid"))
    self.assertEqual(len(events), 5)
    self.assertTupleEqual(events[0], (0, "bar"))
    self.assertTupleEqual(events[1], (1, "tata"))
    self.assertTupleEqual(events[2], (2, "toto"))
    self.assertTupleEqual(events[3], (3, "foo"))
    self.assertTupleEqual(events[4], (4, "titi"))


if __name__ == "__main__":
  tf.test.main()
