# -*- coding: utf-8 -*-

import os

from parameterized import parameterized

import tensorflow as tf
import numpy as np

from opennmt import decoders
from opennmt import encoders
from opennmt import inputters
from opennmt import models
from opennmt.tests import test_util
from opennmt.utils import misc


def _seq2seq_model(training=None):
  model = models.SequenceToSequence(
      inputters.WordEmbedder(16),
      inputters.WordEmbedder(16),
      encoders.SelfAttentionEncoder(2, 16, 4, 32),
      decoders.SelfAttentionDecoder(2, 16, 4, 32))
  params = {}
  if training:
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
    model.initialize(data_config, params=params)
    model.create_variables()
    # Build a dataset for mode.
    if mode == tf.estimator.ModeKeys.PREDICT:
      dataset = model.examples_inputter.make_inference_dataset(
          features_file, batch_size)
    elif mode == tf.estimator.ModeKeys.EVAL:
      dataset = model.examples_inputter.make_evaluation_dataset(
          features_file, labels_file, batch_size)
    elif mode == tf.estimator.ModeKeys.TRAIN:
      dataset = model.examples_inputter.make_training_dataset(
          features_file, labels_file, batch_size)
    # Forward first batch into the model.
    data = iter(dataset).next()
    if mode != tf.estimator.ModeKeys.PREDICT:
      features, labels = data
    else:
      features, labels = data, None
    training = mode == tf.estimator.ModeKeys.TRAIN
    outputs, predictions = model(features, labels=labels, training=training)
    if mode != tf.estimator.ModeKeys.PREDICT:
      loss = model.compute_loss(outputs, labels, training=training)
      if mode == tf.estimator.ModeKeys.EVAL:
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
              lambda x: x.numpy(),
              next(misc.extract_batches(scores)))
          with open(os.devnull, "w") as devnull:
            model.print_score(first_score, stream=devnull)
        except NotImplementedError:
          pass
    else:
      # Check that all prediction heads are returned.
      self.assertIsInstance(predictions, dict)
      if prediction_heads is not None:
        for head in prediction_heads:
          self.assertIn(head, predictions)
      # Check that the prediction can be printed without errors.
      first_prediction = tf.nest.map_structure(
          lambda x: x.numpy(),
          next(misc.extract_batches(predictions)))
      with open(os.devnull, "w") as devnull:
        model.print_prediction(first_prediction, stream=devnull)

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

  def testSequenceToSequenceWithSharedEmbedding(self):
    model = models.SequenceToSequence(
        inputters.WordEmbedder(16),
        inputters.WordEmbedder(16),
        encoders.SelfAttentionEncoder(2, 16, 4, 32),
        decoders.SelfAttentionDecoder(2, 16, 4, 32),
        share_embeddings=models.EmbeddingsSharingLevel.ALL)
    _, _, data_config = self._makeToyEnDeData()
    data_config["target_vocabulary"] = data_config["source_vocabulary"]
    model.initialize(data_config)
    self.assertTrue(model.decoder.initialized)
    model.build(None)
    self.assertEqual(
        model.labels_inputter.embedding.experimental_ref(),
        model.decoder.output_layer.weight.experimental_ref())

  @parameterized.expand([
      [tf.estimator.ModeKeys.EVAL],
      [tf.estimator.ModeKeys.PREDICT]])
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
        params=params)

  @parameterized.expand([["ce"], ["mse"]])
  def testSequenceToSequenceWithGuidedAlignment(self, ga_type):
    model, params = _seq2seq_model(training=True)
    params["guided_alignment_type"] = ga_type
    features_file, labels_file, data_config = self._makeToyEnDeData(with_alignments=True)
    model.initialize(data_config, params=params)
    model.create_variables()
    dataset = model.examples_inputter.make_training_dataset(features_file, labels_file, 16)
    features, labels = next(iter(dataset))
    self.assertIn("alignment", labels)
    outputs, _ = model(features, labels=labels, training=True)
    loss = model.compute_loss(outputs, labels, training=True)
    loss = loss[0] / loss[1]

  def testSequenceToSequenceWithGuidedAlignmentAndWeightedDataset(self):
    model, _ = _seq2seq_model()
    features_file, labels_file, data_config = self._makeToyEnDeData(with_alignments=True)
    model.initialize(data_config)
    with self.assertRaisesRegex(ValueError, "expected to match"):
      model.examples_inputter.make_training_dataset(
          [features_file, features_file], [labels_file, labels_file], 16)
    data_config["train_alignments"] = [
        data_config["train_alignments"], data_config["train_alignments"]]
    model.initialize(data_config)
    dataset = model.examples_inputter.make_training_dataset(
        [features_file, features_file], [labels_file, labels_file], 16)
    self.assertIsInstance(dataset, tf.data.Dataset)

  def testSequenceToSequenceWithReplaceUnknownTarget(self):
    model, params = _seq2seq_model()
    params["replace_unknown_target"] = True
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
        decoders.RNNDecoder(2, 16))
    params = {
        "scheduled_sampling_type": "linear",
        "scheduled_sampling_read_probability": 0.8,
        "scheduled_sampling_k": 0.1
    }
    features_file, labels_file, data_config = self._makeToyEnDeData()
    model.initialize(data_config, params=params)
    dataset = model.examples_inputter.make_training_dataset(features_file, labels_file, 16)
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
    dataset = model.examples_inputter.make_training_dataset(features_file, labels_file, 16)
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
    model.initialize(data_config, params=params)
    function = model.serve_function()
    function.get_concrete_function()

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

  def testLanguageModelInputter(self):
    vocabulary_path = test_util.make_vocab(
        os.path.join(self.get_temp_dir(), "vocab.txt"), ["a", "b", "c"])

    inputter = models.LanguageModelInputter(embedding_size=10)
    inputter.initialize({
        "vocabulary": vocabulary_path,
        "sequence_controls": {"start": True, "end": False}})
    features, labels = self.evaluate(inputter.make_features(tf.constant("a b c")))
    self.assertAllEqual(features["ids"], [1, 3, 4, 5])
    self.assertEqual(features["length"], 4)
    self.assertAllEqual(labels["ids"], [1, 3, 4])
    self.assertAllEqual(labels["ids_out"], [3, 4, 5])
    self.assertEqual(labels["length"], 3)

    # Backward compatibility mode.
    inputter = models.LanguageModelInputter(embedding_size=10)
    inputter.initialize({"vocabulary": vocabulary_path})
    features, labels = self.evaluate(inputter.make_features(tf.constant("a b c")))
    self.assertAllEqual(features["ids"], [3, 4, 5])
    self.assertEqual(features["length"], 3)
    self.assertAllEqual(labels["ids"], [3, 4, 5])
    self.assertAllEqual(labels["ids_out"], [4, 5, 2])
    self.assertEqual(labels["length"], 3)

  def testLanguageModelWithMissingStart(self):
    _, data_config = self._makeToyLMData()
    decoder = decoders.SelfAttentionDecoder(
        2, num_units=16, num_heads=4, ffn_inner_dim=32, num_sources=0)
    model = models.LanguageModel(decoder, embedding_size=16)
    model.initialize(data_config)
    features, _ = model.features_inputter.make_features(tf.constant(""))
    with self.assertRaises(tf.errors.InvalidArgumentError):
      model(features)

  def testLanguageModelWithStartOfSentence(self):
    _, data_config = self._makeToyLMData()
    data_config["sequence_controls"] = dict(start=True, end=False)
    decoder = decoders.SelfAttentionDecoder(
        2, num_units=16, num_heads=4, ffn_inner_dim=32, num_sources=0)
    model = models.LanguageModel(decoder, embedding_size=16)
    model.initialize(data_config, params={"maximum_decoding_length": 1})
    features, _ = model.features_inputter.make_features(tf.constant(""))
    features = tf.nest.map_structure(lambda t: tf.expand_dims(t, 0), features)  # Add batch dim.
    _, predictions = self.evaluate(model(features))
    # Predictions should not include the leading <s>.
    self.assertEqual(predictions["length"][0], 1)
    self.assertTupleEqual(predictions["tokens"].shape, (1, 1))

  @parameterized.expand([
      [tf.estimator.ModeKeys.TRAIN],
      [tf.estimator.ModeKeys.EVAL],
      [tf.estimator.ModeKeys.PREDICT]])
  def testSequenceTagger(self, mode):
    model = models.SequenceTagger(
        inputters.WordEmbedder(10),
        encoders.MeanEncoder(),
        crf_decoding=True)
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

  def testSequenceClassifierWithSelfAttentionEncoder(self):
    # SelfAttentionEncoder does not return a state, so test that the classifier
    # does not crash on this.
    model = models.SequenceClassifier(
        inputters.WordEmbedder(10),
        encoders.SelfAttentionEncoder(num_layers=2, num_units=16, num_heads=4, ffn_inner_dim=32))
    features_file, labels_file, data_config = self._makeToyClassifierData()
    model.initialize(data_config)
    dataset = model.examples_inputter.make_training_dataset(features_file, labels_file, 16)
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
        2, num_units=16, num_heads=4, ffn_inner_dim=32, num_sources=0)
    model = models.LanguageModel(decoder, embedding_size=16)
    model.initialize(data_config)
    model.create_variables()
    self.assertTrue(len(model.trainable_variables) > 0)

  def testInitializeWithDropoutOverride(self):
    model = models.SequenceToSequence(
        inputters.WordEmbedder(16),
        inputters.WordEmbedder(16),
        encoders.SelfAttentionEncoder(2, 16, 4, 32),
        decoders.SelfAttentionDecoder(2, 16, 4, 32))
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
    trainable_variables_ref = set(variable.experimental_ref() for variable in trainable_variables)

    def _assert_layer_not_trainable(layer):
      self.assertFalse(layer.trainable)
      for variable in layer.variables:
        self.assertNotIn(variable.experimental_ref(), trainable_variables_ref)

    _assert_layer_not_trainable(model.decoder.output_layer)
    _assert_layer_not_trainable(model.encoder.layers[0])

  def testTransferWeightsNewVocab(self):

    def _make_model(name, src_vocab, tgt_vocab, random_slots=False):
      model, _ = _seq2seq_model(training=True)
      optimizer = tf.keras.optimizers.Adam()
      data = {}
      data["source_vocabulary"] = test_util.make_data_file(
          os.path.join(self.get_temp_dir(), "%s-src-vocab.txt" % name),
          src_vocab)
      data["target_vocabulary"] = test_util.make_data_file(
          os.path.join(self.get_temp_dir(), "%s-tgt-vocab.txt" % name),
          tgt_vocab)
      model.initialize(data)
      model.create_variables(optimizer=optimizer)
      if random_slots:
        for variable in model.trainable_variables:
          for slot_name in optimizer.get_slot_names():
            slot = optimizer.get_slot(variable, slot_name)
            slot.assign(tf.random.uniform(slot.shape))
      return model, optimizer

    model_a, optimizer_a = _make_model(
        "a", ["a", "b", "c", "d", "e"], ["1", "2", "3", "4", "5", "6"], random_slots=True)
    model_b, optimizer_b = _make_model(
        "b", ["c", "a", "e", "f"], ["1", "3", "2", "6", "7"])
    src_mapping = [2, 0, 4, -1]
    tgt_mapping = [0, 2, 1, 5, -1]

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

    model_a.transfer_weights(model_b, new_optimizer=optimizer_b, optimizer=optimizer_a)
    _check_weight_and_slots(
        lambda model: model.features_inputter.embedding, src_mapping)
    _check_weight_and_slots(
        lambda model: model.labels_inputter.embedding, tgt_mapping)
    _check_weight_and_slots(
        lambda model: model.decoder.output_layer.bias, tgt_mapping)
    _check_weight_and_slots(
        lambda model: model.decoder.output_layer.kernel, tgt_mapping, vocab_axis=1)

  @parameterized.expand([
      [models.TransformerBase()],
      [models.TransformerBaseRelative()],
      [models.TransformerBig()],
      [models.TransformerBigRelative()],
      [models.Transformer(
          inputters.WordEmbedder(32),
          inputters.WordEmbedder(32),
          num_layers=(6, 3),
          num_units=32,
          num_heads=8,
          ffn_inner_dim=64)],
  ])
  def testCTranslate2Spec(self, model):
    try:
      self.assertIsNotNone(model.ctranslate2_spec)
    except ImportError:
      self.skipTest("ctranslate2 module is not available")

  def testTransformerWithDifferentEncoderDecoderLayers(self):
    model = models.Transformer(
        inputters.WordEmbedder(32),
        inputters.WordEmbedder(32),
        num_layers=(6, 3),
        num_units=32,
        num_heads=8,
        ffn_inner_dim=64)
    self.assertLen(model.encoder.layers, 6)
    self.assertLen(model.decoder.layers, 3)

  def testBeamSearchWithMultiSourceEncoder(self):
    shared_vocabulary = test_util.make_vocab(
        os.path.join(self.get_temp_dir(), "vocab.txt"), ["1", "2", "3"])
    data_config = {
        "source_1_vocabulary": shared_vocabulary,
        "source_2_vocabulary": shared_vocabulary,
        "target_vocabulary": shared_vocabulary,
    }
    params = {
        "beam_width": 2,
    }
    model = models.Transformer(
        inputters.ParallelInputter([
            inputters.WordEmbedder(32),
            inputters.WordEmbedder(32)]),
        inputters.WordEmbedder(32),
        num_layers=3,
        num_units=32,
        num_heads=8,
        ffn_inner_dim=64)
    model.initialize(data_config, params=params)
    model.serve_function().get_concrete_function()


if __name__ == "__main__":
  tf.test.main()
