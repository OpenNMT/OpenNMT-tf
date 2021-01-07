import tensorflow as tf

from opennmt.utils import compat


class MiscTest(tf.test.TestCase):
    def testTFSupports(self):
        self.assertTrue(compat.tf_supports("data"))
        self.assertTrue(compat.tf_supports("data.Dataset"))
        self.assertFalse(compat.tf_supports("data.UnknwonClass"))
        self.assertFalse(compat.tf_supports("unknown_module"))


if __name__ == "__main__":
    tf.test.main()
