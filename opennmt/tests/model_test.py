import os

import numpy as np
import tensorflow as tf

from parameterized import parameterized

from opennmt import decoders, encoders, inputters, models
from opennmt.optimizers.utils import make_optimizer
from opennmt.tests import test_util
from opennmt.utils import misc


def _seq2seq_model(training=None, shared_embeddings=False):
    model = models.SequenceToSequence(
        inputters.WordEmbedder(16),
        inputters.WordEmbedder(16),
        encoders.SelfAttentionEncoder(2, 16, 4, 32),
        decoders.SelfAttentionDecoder(2, 16, 4, 32),
        share_embeddings=(
            models.sequence_to_sequence.EmbeddingsSharingLevel.ALL
            if shared_embeddings
            else models.EmbeddingsSharingLevel.NONE
        ),
    )
    params = {}
    if training:
        params["optimizer"] = "SGD"
        params["learning_rate"] = 0.1
    return model, params


class ModelTest(tf.test.TestCase):
    def _makeToyEnDeData(self, with_alignments=False, with_weights=False):
        data_config = {}
        features_file = test_util.make_data_file(
            os.path.join(self.get_temp_dir(), "src.txt"),
            [
                "Parliament Does Not Support Amendment Freeing Tymoshenko",
                "Today , the Ukraine parliament dismissed , within the Code of Criminal Procedure "
                "amendment , the motion to revoke an article based on which the opposition leader "
                ", Yulia Tymoshenko , was sentenced .",
                "The amendment that would lead to freeing the imprisoned former Prime Minister was "
                "revoked during second reading of the proposal for mitigation of sentences for "
                "economic offences .",
            ],
        )
        labels_file = test_util.make_data_file(
            os.path.join(self.get_temp_dir(), "tgt.txt"),
            [
                "Keine befreiende Novelle für Tymoshenko durch das Parlament",
                "Das ukrainische Parlament verweigerte heute den Antrag , im Rahmen einer Novelle "
                "des Strafgesetzbuches denjenigen Paragrafen abzuschaffen , auf dessen Grundlage "
                "die Oppositionsführerin Yulia Timoshenko verurteilt worden war .",
                "Die Neuregelung , die den Weg zur Befreiung der inhaftierten Expremierministerin "
                "hätte ebnen können , lehnten die Abgeordneten bei der zweiten Lesung des Antrags "
                "auf Milderung der Strafen für wirtschaftliche Delikte ab .",
            ],
        )
        data_config["source_vocabulary"] = test_util.make_vocab_from_file(
            os.path.join(self.get_temp_dir(), "src_vocab.txt"), features_file
        )
        data_config["target_vocabulary"] = test_util.make_vocab_from_file(
            os.path.join(self.get_temp_dir(), "tgt_vocab.txt"), labels_file
        )
        if with_alignments:
            # Dummy and incomplete alignments.
            data_config["train_alignments"] = test_util.make_data_file(
                os.path.join(self.get_temp_dir(), "alignments.txt"),
                [
                    "0-0 1-0 2-2 3-4 4-4 5-6",
                    "0-1 1-1 1-3 2-3 4-4",
                    "0-0 1-0 2-2 3-4 4-4 5-6",
                ],
            )
        if with_weights:
            data_config["example_weights"] = test_util.make_data_file(
                os.path.join(self.get_temp_dir(), "weights.txt"), ["0.6", "1", "1e-2"]
            )
        return features_file, labels_file, data_config

    def _makeToyLMData(self):
        features_file, _, data_config = self._makeToyEnDeData()
        return features_file, {"vocabulary": data_config["source_vocabulary"]}

    def _makeToyTaggerData(self):
        data_config = {}
        features_file = test_util.make_data_file(
            os.path.join(self.get_temp_dir(), "src.txt"),
            ["M . Smith went to Washington .", "I live in New Zealand ."],
        )
        labels_file = test_util.make_data_file(
            os.path.join(self.get_temp_dir(), "labels.txt"),
            ["B-PER I-PER E-PER O O S-LOC O", "O O O B-LOC E-LOC O"],
        )
        data_config["source_vocabulary"] = test_util.make_vocab_from_file(
            os.path.join(self.get_temp_dir(), "src_vocab.txt"), features_file
        )
        data_config["target_vocabulary"] = test_util.make_data_file(
            os.path.join(self.get_temp_dir(), "labels_vocab.txt"),
            [
                "O",
                "B-LOC",
                "I-LOC",
                "E-LOC",
                "S-LOC",
                "B-PER",
                "I-PER",
                "E-PER",
                "S-PER",
            ],
        )
        return features_file, labels_file, data_config

    def _makeToyClassifierData(self):
        data_config = {}
        features_file = test_util.make_data_file(
            os.path.join(self.get_temp_dir(), "src.txt"),
            [
                "This product was not good at all , it broke on the first use !",
                "Perfect , it does everything I need .",
                "How do I change the battery ?",
            ],
        )
        labels_file = test_util.make_data_file(
            os.path.join(self.get_temp_dir(), "labels.txt"),
            ["negative", "positive", "neutral"],
        )
        data_config["source_vocabulary"] = test_util.make_vocab_from_file(
            os.path.join(self.get_temp_dir(), "src_vocab.txt"), features_file
        )
        data_config["target_vocabulary"] = test_util.make_data_file(
            os.path.join(self.get_temp_dir(), "labels_vocab.txt"),
            ["negative", "positive", "neutral"],
        )
        return features_file, labels_file, data_config

    def _testGenericModel(
        self,
        model,
        mode,
        features_file,
        labels_file=None,
        data_config=None,
        batch_size=16,
        prediction_heads=None,
        metrics=None,
        params=None,
    ):
        # Mainly test that the code does not throw.
        if params is None:
            params = model.auto_config()["params"]
        if data_config is None:
            data_config = {}
        model.initialize(data_config, params=params)
        model.create_variables()

        # Build a dataset for mode.
        if mode == tf.estimator.ModeKeys.PREDICT:
            dataset = model.examples_inputter.make_inference_dataset(
                features_file, batch_size
            )

            features = next(iter(dataset))
            predictions = model.infer(features)

            # Check that all prediction heads are returned.
            self.assertIsInstance(predictions, dict)
            if prediction_heads is not None:
                for head in prediction_heads:
                    self.assertIn(head, predictions)
            # Check that the prediction can be printed without errors.
            first_prediction = tf.nest.map_structure(
                lambda x: x.numpy(), next(misc.extract_batches(predictions))
            )
            with open(os.devnull, "w") as devnull:
                model.print_prediction(first_prediction, stream=devnull)

        elif mode == tf.estimator.ModeKeys.EVAL:
            dataset = model.examples_inputter.make_evaluation_dataset(
                features_file, labels_file, batch_size
            )

            features, labels = model.split_features_labels(next(iter(dataset)))
            loss, predictions = model.evaluate(features, labels)

            # Check that returned evaluation metrics are expected.
            eval_metrics = model.get_metrics()
            if eval_metrics is not None:
                model.update_metrics(eval_metrics, predictions, labels)
                for metric in metrics:
                    self.assertIn(metric, eval_metrics)
            try:
                # Check that scores can be computed and printed without errors.
                scores = model.score(features, labels)
                first_score = tf.nest.map_structure(
                    lambda x: x.numpy(), next(misc.extract_batches(scores))
                )
                with open(os.devnull, "w") as devnull:
                    model.print_score(first_score, stream=devnull)
            except NotImplementedError:
                pass

        elif mode == tf.estimator.ModeKeys.TRAIN:
            dataset = model.examples_inputter.make_training_dataset(
                features_file, labels_file, batch_size
            )

            features, labels = model.split_features_labels(next(iter(dataset)))
            outputs, _ = model(features, labels, training=True)
            _ = model.compute_loss(outputs, labels, training=True)

    @parameterized.expand(
        [
            [tf.estimator.ModeKeys.TRAIN],
            [tf.estimator.ModeKeys.EVAL],
            [tf.estimator.ModeKeys.PREDICT],
        ]
    )
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
            params=params,
        )

    @parameterized.expand(
        [
            (models.EmbeddingsSharingLevel.ALL, True, True, True),
            (models.EmbeddingsSharingLevel.AUTO, True, True, True),
            (models.EmbeddingsSharingLevel.AUTO, False, False, True),
        ]
    )
    def testSequenceToSequenceWithSharedEmbedding(
        self, share_embeddings, reuse_vocab, input_is_shared, target_is_shared
    ):
        model = models.SequenceToSequence(
            inputters.WordEmbedder(16),
            inputters.WordEmbedder(16),
            encoders.SelfAttentionEncoder(2, 16, 4, 32),
            decoders.SelfAttentionDecoder(2, 16, 4, 32),
            share_embeddings=share_embeddings,
        )
        _, _, data_config = self._makeToyEnDeData()
        if reuse_vocab:
            data_config["target_vocabulary"] = data_config["source_vocabulary"]
        model.initialize(data_config)
        model.create_variables()

        self.assertEqual(
            model.features_inputter.embedding.ref()
            == model.labels_inputter.embedding.ref(),
            input_is_shared,
        )
        self.assertEqual(
            model.labels_inputter.embedding.ref()
            == model.decoder.output_layer.kernel.ref(),
            target_is_shared,
        )

    @parameterized.expand(
        [[tf.estimator.ModeKeys.EVAL], [tf.estimator.ModeKeys.PREDICT]]
    )
    def testSequenceToSequenceWithInGraphTokenizer(self, mode):
        model, params = _seq2seq_model(mode)
        features_file, labels_file, data_config = self._makeToyEnDeData()
        tokenization_config = {"type": "SpaceTokenizer"}
        data_config["source_tokenization"] = tokenization_config
        data_config["target_tokenization"] = tokenization_config
        self._testGenericModel(
            model,
            mode,
            features_file,
            labels_file,
            data_config,
            prediction_heads=["text", "log_probs"],
            params=params,
        )

    @parameterized.expand([["ce"], ["mse"]])
    def testSequenceToSequenceWithGuidedAlignment(self, ga_type):
        model, params = _seq2seq_model(training=True)
        params["guided_alignment_type"] = ga_type
        features_file, labels_file, data_config = self._makeToyEnDeData(
            with_alignments=True
        )
        model.initialize(data_config, params=params)
        model.create_variables()
        dataset = model.examples_inputter.make_training_dataset(
            features_file, labels_file, 16
        )
        features, labels = next(iter(dataset))
        self.assertIn("alignment", labels)
        outputs, _ = model(features, labels=labels, training=True)
        loss = model.compute_loss(outputs, labels, training=True)
        loss = loss[0] / loss[1]

    @test_util.run_with_mixed_precision
    def testSequenceToSequenceWithGuidedAlignmentMixedPrecision(self):
        model, params = _seq2seq_model(training=True)
        params["guided_alignment_type"] = "ce"
        features_file, labels_file, data_config = self._makeToyEnDeData(
            with_alignments=True
        )
        model.initialize(data_config, params=params)
        model.create_variables()
        dataset = model.examples_inputter.make_training_dataset(
            features_file, labels_file, 16
        )
        features, labels = next(iter(dataset))
        outputs, _ = model(features, labels=labels, training=True)
        model.compute_loss(outputs, labels, training=True)

    def testSequenceToSequenceWithGuidedAlignmentAndWeightedDataset(self):
        model, _ = _seq2seq_model()
        features_file, labels_file, data_config = self._makeToyEnDeData(
            with_alignments=True
        )
        model.initialize(data_config)
        with self.assertRaisesRegex(ValueError, "expected to match"):
            model.examples_inputter.make_training_dataset(
                [features_file, features_file], [labels_file, labels_file], 16
            )
        data_config["train_alignments"] = [
            data_config["train_alignments"],
            data_config["train_alignments"],
        ]
        model.initialize(data_config)
        dataset = model.examples_inputter.make_training_dataset(
            [features_file, features_file], [labels_file, labels_file], 16
        )
        self.assertIsInstance(dataset, tf.data.Dataset)

    def testSequenceToSequenceWithWeightedExamples(self):
        model, params = _seq2seq_model(training=True)
        features_file, labels_file, data_config = self._makeToyEnDeData(
            with_weights=True
        )
        model.initialize(data_config, params=params)
        dataset = model.examples_inputter.make_training_dataset(
            features_file, labels_file, 16
        )
        features, labels = next(iter(dataset))
        self.assertIn("weight", labels)
        outputs, _ = model(features, labels=labels, training=True)
        weighted_loss, _, _ = model.compute_loss(outputs, labels, training=True)
        labels.pop("weight")
        default_loss, _, _ = model.compute_loss(outputs, labels, training=True)
        self.assertNotEqual(weighted_loss, default_loss)

    def testSequenceToSequenceWithReplaceUnknownTarget(self):
        model, params = _seq2seq_model()
        params["replace_unknown_target"] = True
        params["beam_width"] = 2
        features_file, labels_file, data_config = self._makeToyEnDeData()
        data_config["source_sequence_controls"] = {"start": True, "end": True}
        model.initialize(data_config, params=params)
        dataset = model.examples_inputter.make_inference_dataset(features_file, 16)
        features = next(iter(dataset))
        _, predictions = model(features)

    def testSequenceToSequenceWithNoisyDecoding(self):
        model, params = _seq2seq_model()
        params["maximum_decoding_length"] = 20
        params["beam_width"] = 2
        params["decoding_noise"] = [
            {"dropout": 0.1},
            {"replacement": [0.1, "<unk>"]},
            {"permutation": 3},
        ]
        features_file, labels_file, data_config = self._makeToyEnDeData()
        model.initialize(data_config, params=params)
        dataset = model.examples_inputter.make_inference_dataset(features_file, 16)
        features = next(iter(dataset))
        _, predictions = model(features)

    def testSequenceToSequenceWithScheduledSampling(self):
        model = models.SequenceToSequence(
            inputters.WordEmbedder(16),
            inputters.WordEmbedder(16),
            encoders.SelfAttentionEncoder(2, 16, 4, 32),
            decoders.RNNDecoder(2, 16),
        )
        params = {
            "scheduled_sampling_type": "linear",
            "scheduled_sampling_read_probability": 0.8,
            "scheduled_sampling_k": 0.1,
        }
        features_file, labels_file, data_config = self._makeToyEnDeData()
        model.initialize(data_config, params=params)
        dataset = model.examples_inputter.make_training_dataset(
            features_file, labels_file, 16
        )
        features, labels = next(iter(dataset))
        with self.assertRaises(ValueError):
            model(features, labels=labels, training=True)  # step argument is required.
        outputs, _ = model(features, labels=labels, training=True, step=10)
        self.assertEqual(outputs["logits"].shape[1], labels["ids"].shape[1])

    def testSequenceToSequenceWithContrastiveLearning(self):
        model, params = _seq2seq_model()
        params["contrastive_learning"] = True
        features_file, labels_file, data_config = self._makeToyEnDeData()
        model.initialize(data_config, params=params)
        dataset = model.examples_inputter.make_training_dataset(
            features_file, labels_file, 16
        )
        features, labels = next(iter(dataset))
        self.assertIn("noisy_ids", labels)
        self.assertIn("noisy_ids_out", labels)
        self.assertIn("noisy_length", labels)
        outputs, _ = model(features, labels=labels, training=True)
        self.assertIn("noisy_logits", outputs)
        loss = model.compute_loss(outputs, labels, training=True)
        self.assertGreaterEqual(self.evaluate(loss), 0)

    def testSequenceToSequenceServing(self):
        # Test that serving features can be forwarded into the model.
        _, _, data_config = self._makeToyEnDeData()
        model, params = _seq2seq_model()
        params["beam_width"] = 4
        model.initialize(data_config, params=params)
        function = model.serve_function()
        function.get_concrete_function()

    @test_util.run_with_mixed_precision
    def testRNNWithMixedPrecision(self):
        features_file, labels_file, data_config = self._makeToyEnDeData()
        model = models.LuongAttention()
        model.initialize(data_config)
        dataset = model.examples_inputter.make_training_dataset(
            features_file, labels_file, 16
        )
        features, labels = next(iter(dataset))
        outputs, _ = model(features, labels=labels, training=True)
        self.assertEqual(outputs["logits"].dtype, tf.float16)
        self.assertEqual(outputs["attention"].dtype, tf.float16)

    @parameterized.expand(
        [
            [tf.estimator.ModeKeys.TRAIN],
            [tf.estimator.ModeKeys.EVAL],
            [tf.estimator.ModeKeys.PREDICT],
        ]
    )
    def testLanguageModel(self, mode):
        # Mainly test that the code does not throw.
        decoder = decoders.SelfAttentionDecoder(
            2, num_units=16, num_heads=4, ffn_inner_dim=32, num_sources=0
        )
        model = models.LanguageModel(decoder, embedding_size=16)
        features_file, data_config = self._makeToyLMData()
        params = {"optimizer": "SGD", "learning_rate": 0.1}
        self._testGenericModel(
            model,
            mode,
            features_file,
            data_config=data_config,
            batch_size=1 if mode == tf.estimator.ModeKeys.PREDICT else 16,
            prediction_heads=["tokens", "length"],
            params=params,
        )

    def testLanguageModelServing(self):
        _, data_config = self._makeToyLMData()
        decoder = decoders.SelfAttentionDecoder(
            2, num_units=16, num_heads=4, ffn_inner_dim=32, num_sources=0
        )
        model = models.LanguageModel(decoder, embedding_size=16)
        model.initialize(data_config)
        function = model.serve_function()
        function.get_concrete_function()

    def testLanguageModelInputter(self):
        vocabulary_path = test_util.make_vocab(
            os.path.join(self.get_temp_dir(), "vocab.txt"), ["a", "b", "c"]
        )

        inputter = models.LanguageModelInputter(embedding_size=10)
        inputter.initialize(
            {
                "vocabulary": vocabulary_path,
                "sequence_controls": {"start": True, "end": False},
            }
        )
        features = inputter.make_features("a b c")
        self.assertAllEqual(features["ids"], [1, 3, 4])
        self.assertAllEqual(features["ids_out"], [3, 4, 5])
        self.assertEqual(features["length"], 3)

        # Inference mode.
        inputter.inference = True
        features = inputter.make_features("a b c")
        self.assertAllEqual(features["ids"], [1, 3, 4, 5])
        self.assertEqual(features["length"], 4)
        inputter.inference = False

        # Backward compatibility mode.
        inputter = models.LanguageModelInputter(embedding_size=10)
        inputter.initialize({"vocabulary": vocabulary_path})
        features = inputter.make_features("a b c")
        self.assertAllEqual(features["ids"], [3, 4, 5])
        self.assertAllEqual(features["ids_out"], [4, 5, 2])
        self.assertEqual(features["length"], 3)

    def testLanguageModelWithMissingStart(self):
        _, data_config = self._makeToyLMData()
        decoder = decoders.SelfAttentionDecoder(
            2, num_units=16, num_heads=4, ffn_inner_dim=32, num_sources=0
        )
        model = models.LanguageModel(decoder, embedding_size=16)
        model.initialize(data_config)
        features = model.features_inputter.make_features("")
        with self.assertRaises(tf.errors.InvalidArgumentError):
            model(features)

    def testLanguageModelWithStartOfSentence(self):
        _, data_config = self._makeToyLMData()
        data_config["sequence_controls"] = dict(start=True, end=False)
        decoder = decoders.SelfAttentionDecoder(
            2, num_units=16, num_heads=4, ffn_inner_dim=32, num_sources=0
        )
        model = models.LanguageModel(decoder, embedding_size=16)
        model.initialize(data_config, params={"maximum_decoding_length": 1})
        features = model.features_inputter.make_features("")
        features = tf.nest.map_structure(
            lambda t: tf.expand_dims(t, 0), features
        )  # Add batch dim.
        _, predictions = self.evaluate(model(features))
        # Predictions should not include the leading <s>.
        self.assertEqual(predictions["length"][0], 1)
        self.assertTupleEqual(predictions["tokens"].shape, (1, 1))

    def testLanguageModelBucketing(self):
        features_file, data_config = self._makeToyLMData()
        decoder = decoders.SelfAttentionDecoder(
            2, num_units=16, num_heads=4, ffn_inner_dim=32, num_sources=0
        )
        model = models.LanguageModel(decoder, embedding_size=16)
        model.initialize(data_config)
        model.features_inputter.make_inference_dataset(
            features_file, batch_size=8, length_bucket_width=1
        )

    def testLanguageModelBatchAutotune(self):
        features_file, data_config = self._makeToyLMData()
        decoder = decoders.SelfAttentionDecoder(
            2, num_units=16, num_heads=4, ffn_inner_dim=32, num_sources=0
        )
        model = models.LanguageModel(decoder, embedding_size=16)
        model.initialize(data_config)
        model.examples_inputter.make_training_dataset(
            features_file,
            None,
            batch_size=8,
            length_bucket_width=1,
            maximum_features_length=50,
            batch_autotune_mode=True,
        )

    @parameterized.expand(
        [
            [tf.estimator.ModeKeys.TRAIN],
            [tf.estimator.ModeKeys.EVAL],
            [tf.estimator.ModeKeys.PREDICT],
        ]
    )
    def testSequenceTagger(self, mode):
        model = models.SequenceTagger(
            inputters.WordEmbedder(10), encoders.MeanEncoder(), crf_decoding=True
        )
        features_file, labels_file, data_config = self._makeToyTaggerData()
        data_config["tagging_scheme"] = "bioes"
        params = {"optimizer": "SGD", "learning_rate": 0.1}
        self._testGenericModel(
            model,
            mode,
            features_file,
            labels_file,
            data_config,
            prediction_heads=["tags", "length"],
            metrics=["accuracy", "precision", "recall", "f1"],
            params=params,
        )

    @parameterized.expand(
        [
            [tf.estimator.ModeKeys.TRAIN],
            [tf.estimator.ModeKeys.EVAL],
            [tf.estimator.ModeKeys.PREDICT],
        ]
    )
    def testSequenceClassifier(self, mode):
        model = models.SequenceClassifier(
            inputters.WordEmbedder(10), encoders.MeanEncoder()
        )
        features_file, labels_file, data_config = self._makeToyClassifierData()
        params = {"optimizer": "SGD", "learning_rate": 0.1}
        self._testGenericModel(
            model,
            mode,
            features_file,
            labels_file,
            data_config,
            prediction_heads=["classes"],
            metrics=["accuracy"],
            params=params,
        )

    def testSequenceClassifierWithSelfAttentionEncoder(self):
        # SelfAttentionEncoder does not return a state, so test that the classifier
        # does not crash on this.
        model = models.SequenceClassifier(
            inputters.WordEmbedder(10),
            encoders.SelfAttentionEncoder(
                num_layers=2, num_units=16, num_heads=4, ffn_inner_dim=32
            ),
        )
        features_file, labels_file, data_config = self._makeToyClassifierData()
        model.initialize(data_config)
        dataset = model.examples_inputter.make_training_dataset(
            features_file, labels_file, 16
        )
        features, labels = iter(dataset).next()
        model(features, labels, training=True)

    def testCreateVariables(self):
        _, _, data_config = self._makeToyEnDeData()
        model, params = _seq2seq_model()
        model.initialize(data_config, params=params)
        model.create_variables()
        self.assertTrue(len(model.trainable_variables) > 0)

    def testCreateVariablesLanguageModel(self):
        _, data_config = self._makeToyLMData()
        decoder = decoders.SelfAttentionDecoder(
            2, num_units=16, num_heads=4, ffn_inner_dim=32, num_sources=0
        )
        model = models.LanguageModel(decoder, embedding_size=16)
        model.initialize(data_config)
        model.create_variables()
        self.assertTrue(len(model.trainable_variables) > 0)

    def testInitializeWithDropoutOverride(self):
        model = models.SequenceToSequence(
            inputters.WordEmbedder(16),
            inputters.WordEmbedder(16),
            encoders.SelfAttentionEncoder(2, 16, 4, 32),
            decoders.SelfAttentionDecoder(2, 16, 4, 32),
        )
        self.assertEqual(model.encoder.dropout, 0.1)
        _, _, data_config = self._makeToyClassifierData()
        params = dict(dropout=0.3)
        model.initialize(data_config, params=params)
        self.assertEqual(model.encoder.dropout, 0.3)

    def testFreezeLayers(self):
        model, _ = _seq2seq_model(training=True)
        params = {"freeze_layers": ["decoder/output_layer", "encoder/layers/0"]}
        _, _, data_config = self._makeToyEnDeData()
        model.initialize(data_config, params=params)
        model.create_variables()
        trainable_variables = model.trainable_variables
        self.assertNotEmpty(trainable_variables)
        trainable_variables_ref = set(
            variable.ref() for variable in trainable_variables
        )

        def _assert_layer_not_trainable(layer):
            self.assertFalse(layer.trainable)
            for variable in layer.variables:
                self.assertNotIn(variable.ref(), trainable_variables_ref)

        _assert_layer_not_trainable(model.decoder.output_layer)
        _assert_layer_not_trainable(model.encoder.layers[0])
        self.assertEqual(model.encoder.layers[0].ffn.output_dropout, 0)
        self.assertEqual(model.encoder.layers[0].self_attention.output_dropout, 0)

    @parameterized.expand([[True], [False]])
    def testTransferWeightsNewVocab(self, shared_embeddings):
        def _make_model(name, src_vocab, tgt_vocab, random_slots=False):
            model, _ = _seq2seq_model(
                training=True, shared_embeddings=shared_embeddings
            )
            optimizer = make_optimizer("Adam", 0.001)
            data = {}
            data["source_vocabulary"] = test_util.make_data_file(
                os.path.join(self.get_temp_dir(), "%s-src-vocab.txt" % name), src_vocab
            )
            data["target_vocabulary"] = test_util.make_data_file(
                os.path.join(self.get_temp_dir(), "%s-tgt-vocab.txt" % name), tgt_vocab
            )
            model.initialize(data)
            model.create_variables(optimizer=optimizer)
            if random_slots:
                for variable in model.trainable_variables:
                    for slot_name in optimizer.get_slot_names():
                        slot = optimizer.get_slot(variable, slot_name)
                        slot.assign(tf.random.uniform(slot.shape))
            return model, optimizer

        cur_src_vocab = ["a", "b", "c", "d", "e"]
        new_src_vocab = ["c", "a", "e", "f"]
        src_mapping = [2, 0, 4, -1]

        if shared_embeddings:
            cur_tgt_vocab = cur_src_vocab
            new_tgt_vocab = new_src_vocab
            tgt_mapping = src_mapping
        else:
            cur_tgt_vocab = ["1", "2", "3", "4", "5", "6"]
            new_tgt_vocab = ["1", "3", "2", "6", "7"]
            tgt_mapping = [0, 2, 1, 5, -1]

        model_a, optimizer_a = _make_model(
            "a",
            cur_src_vocab,
            cur_tgt_vocab,
            random_slots=True,
        )
        model_b, optimizer_b = _make_model(
            "b",
            new_src_vocab,
            new_tgt_vocab,
        )

        def _check_weight(weight_a, weight_b, mapping, vocab_axis=0):
            weight_a = self.evaluate(weight_a)
            weight_b = self.evaluate(weight_b)
            if vocab_axis != 0:
                perm = list(range(len(weight_a.shape)))
                perm[0], perm[vocab_axis] = perm[vocab_axis], perm[0]
                weight_a = np.transpose(weight_a, axes=perm)
                weight_b = np.transpose(weight_b, axes=perm)
            self.assertEqual(weight_b.shape[0], len(mapping) + 1)
            for index_b, index_a in enumerate(mapping):
                if index_a >= 0:
                    self.assertAllEqual(weight_b[index_b], weight_a[index_a])

        def _check_weight_and_slots(weight_fn, mapping, vocab_axis=0):
            weight_a = weight_fn(model_a)
            weight_b = weight_fn(model_b)
            _check_weight(weight_a, weight_b, mapping, vocab_axis=vocab_axis)
            for slot_name in optimizer_b.get_slot_names():
                slot_a = optimizer_a.get_slot(weight_a, slot_name)
                slot_b = optimizer_b.get_slot(weight_b, slot_name)
                _check_weight(slot_a, slot_b, mapping, vocab_axis=vocab_axis)

        model_a.transfer_weights(
            model_b, new_optimizer=optimizer_b, optimizer=optimizer_a
        )

        self.assertEqual(len(model_a.variables), len(model_b.variables))
        self.assertEqual(len(optimizer_a.variables()), len(optimizer_b.variables()))

        _check_weight_and_slots(
            lambda model: model.features_inputter.embedding, src_mapping
        )
        _check_weight_and_slots(
            lambda model: model.decoder.output_layer.bias, tgt_mapping
        )

        if shared_embeddings:
            self.assertIs(
                model_b.features_inputter.embedding,
                model_b.labels_inputter.embedding,
            )
            self.assertIs(
                model_b.features_inputter.embedding,
                model_b.decoder.output_layer.kernel,
            )
        else:
            _check_weight_and_slots(
                lambda model: model.labels_inputter.embedding, tgt_mapping
            )
            _check_weight_and_slots(
                lambda model: model.decoder.output_layer.bias, tgt_mapping
            )
            _check_weight_and_slots(
                lambda model: model.decoder.output_layer.kernel,
                tgt_mapping,
                vocab_axis=1,
            )

    def testTransformerWithDifferentEncoderDecoderLayers(self):
        model = models.Transformer(
            inputters.WordEmbedder(32),
            inputters.WordEmbedder(32),
            num_layers=(6, 3),
            num_units=32,
            num_heads=8,
            ffn_inner_dim=64,
        )
        self.assertLen(model.encoder.layers, 6)
        self.assertLen(model.decoder.layers, 3)

    def testTransformerNoOutputBias(self):
        _, _, data_config = self._makeToyEnDeData()
        model = models.Transformer(output_layer_bias=False)
        model.initialize(data_config)
        self.assertFalse(model.decoder.output_layer.use_bias)

    def testBeamSearchWithMultiSourceEncoder(self):
        shared_vocabulary = test_util.make_vocab(
            os.path.join(self.get_temp_dir(), "vocab.txt"), ["1", "2", "3"]
        )
        data_config = {
            "source_1_vocabulary": shared_vocabulary,
            "source_2_vocabulary": shared_vocabulary,
            "target_vocabulary": shared_vocabulary,
        }
        params = {
            "beam_width": 2,
        }
        model = models.Transformer(
            inputters.ParallelInputter(
                [inputters.WordEmbedder(32), inputters.WordEmbedder(32)]
            ),
            inputters.WordEmbedder(32),
            num_layers=3,
            num_units=32,
            num_heads=8,
            ffn_inner_dim=64,
        )
        model.initialize(data_config, params=params)
        model.serve_function().get_concrete_function()

    @parameterized.expand([[True], [False]])
    def testTrainModelOnBatch(self, jit_compile):
        _, _, data_config = self._makeToyEnDeData()
        optimizer = make_optimizer("Adam", 0.001)
        model = models.TransformerTiny()
        model.initialize(data_config)
        model.set_jit_compile(jit_compile)
        features = model.features_inputter.make_features(
            ["hello world !", "how are you ?"]
        )
        labels = model.labels_inputter.make_features(
            ["hallo welt !", "wie geht es dir ?"]
        )
        loss1 = model.train(features, labels, optimizer)
        loss2 = model.train(features, labels, optimizer)
        self.assertLess(loss2, loss1)

    def testLossNormalization(self):
        model = models.TransformerTiny()

        _, _, data_config = self._makeToyEnDeData()
        params = model.auto_config()["params"]
        params["dropout"] = 0

        model.initialize(data_config, params=params)
        optimizer = model.get_optimizer()

        features = model.features_inputter.make_features(
            ["hello world !", "how are you ?"]
        )
        labels = model.labels_inputter.make_features(
            ["hallo welt !", "wie geht es dir ?"]
        )

        normalized_loss, _ = model.compute_gradients(features, labels, optimizer)
        cumulated_loss, _, sample_size = model.compute_gradients(
            features, labels, optimizer, normalize_loss=False
        )

        # sample_size should be the sum of the target lengths including EOS.
        self.assertEqual(sample_size, 4 + 6)
        self.assertAllClose(cumulated_loss / sample_size, normalized_loss)


if __name__ == "__main__":
    tf.test.main()
