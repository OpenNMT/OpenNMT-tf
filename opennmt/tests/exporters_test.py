import os

import tensorflow as tf

from opennmt import models
from opennmt.tests import test_util
from opennmt.utils import exporters


class ExportersTest(tf.test.TestCase):
    def testCheckpointExport(self):
        vocab = test_util.make_vocab(
            os.path.join(self.get_temp_dir(), "vocab.txt"), ["a", "b", "c"]
        )

        model = models.TransformerBase()
        model.initialize(dict(source_vocabulary=vocab, target_vocabulary=vocab))
        model.create_variables()
        original_embedding = model.features_inputter.embedding.numpy()

        export_dir = self.get_temp_dir()
        exporter = exporters.CheckpointExporter()
        exporter.export(model, export_dir)

        model = models.TransformerBase()
        model.initialize(dict(source_vocabulary=vocab, target_vocabulary=vocab))
        model.create_variables()

        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint.restore(os.path.join(export_dir, "ckpt"))

        restored_embedding = model.features_inputter.embedding.numpy()
        self.assertAllEqual(restored_embedding, original_embedding)


if __name__ == "__main__":
    tf.test.main()
