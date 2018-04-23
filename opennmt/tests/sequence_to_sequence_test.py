import tensorflow as tf

from opennmt.models import sequence_to_sequence


class SequenceToSequenceTest(tf.test.TestCase):

  def testReplaceUnknownTarget(self):
    target_tokens = [
      ["Hello", "world", "!", "", "", ""],
      ["<unk>", "name", "is", "<unk>", ".", ""]]
    source_tokens = [
      ["Bonjour", "le", "monde", "!", ""],
      ["Mon", "nom", "est", "Max", "."]]
    attention = [
      [[0.9, 0.1, 0.0, 0.0, 0.0],
       [0.2, 0.1, 0.7, 0.0, 0.0],
       [0.0, 0.1, 0.1, 0.8, 0.0],
       [0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0, 0.0]],
      [[0.8, 0.1, 0.1, 0.0, 0.0],
       [0.1, 0.9, 0.0, 0.0, 0.0],
       [0.0, 0.1, 0.8, 0.1, 0.0],
       [0.1, 0.1, 0.2, 0.6, 0.0],
       [0.0, 0.1, 0.1, 0.3, 0.5],
       [0.0, 0.0, 0.0, 0.0, 0.0]]]
    replaced_target_tokens = sequence_to_sequence.replace_unknown_target(
        target_tokens,
        source_tokens,
        attention,
        unknown_token="<unk>")
    with self.test_session() as sess:
      replaced_target_tokens = sess.run(replaced_target_tokens)
      self.assertNotIn(b"<unk>", replaced_target_tokens.flatten().tolist())
      self.assertListEqual(
          [b"Hello", b"world", b"!", b"", b"", b""], replaced_target_tokens[0].tolist())
      self.assertListEqual(
          [b"Mon", b"name", b"is", b"Max", b".", b""], replaced_target_tokens[1].tolist())


if __name__ == "__main__":
  tf.test.main()
