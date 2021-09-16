import inspect

import tensorflow as tf

import opennmt


class APITest(tf.test.TestCase):
    def testSubmodules(self):
        def _check(module):
            self.assertTrue(inspect.ismodule(module))

        _check(opennmt.data)
        _check(opennmt.decoders)
        _check(opennmt.encoders)
        _check(opennmt.inputters)
        _check(opennmt.layers)
        _check(opennmt.models)
        _check(opennmt.optimizers)
        _check(opennmt.schedules)
        _check(opennmt.tokenizers)
        _check(opennmt.utils)


if __name__ == "__main__":
    tf.test.main()
