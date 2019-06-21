"""Evaluation related classes and functions."""

import abc
import os
import six

import tensorflow as tf

from opennmt.data import dataset as dataset_lib
from opennmt.utils import misc


class Evaluator(object):
  """Model evaluator."""

  def __init__(self,
               model,
               features_file,
               labels_file,
               batch_size,
               params=None,
               scorers=None,
               save_predictions=False,
               eval_dir=None):
    """Initializes the evaluator.

    Args:
      model: A :class:`opennmt.models.model.Model` to evaluate.
      features_file: Path to the evaluation features.
      labels_file: Path to the evaluation labels.
      batch_size: The evaluation batch size.
      params: Dictionary of hyperparameters.
      scorers: A list of scorers, callables taking the path to the reference and
        the hypothesis and return one or more scores.
      save_predictions: Save evaluation predictions to a file. This is ``True``
        when :obj:`external_evaluator` is set.
      eval_dir: Directory where predictions can be saved.

    Raises:
      ValueError: If predictions should be saved but the model is not compatible.
      ValueError: If predictions should be saved but :obj:`eval_dir` is ``None``.
    """
    if params is None:
      params = {}
    if scorers is None:
      scorers = []
    if scorers:
      save_predictions = True
    if save_predictions:
      if model.unsupervised:
        raise ValueError("This model does not support saving evaluation predictions")
      if eval_dir is None:
        raise ValueError("Saving evaluation predictions requires eval_dir to be set")
      if not tf.io.gfile.exists(eval_dir):
        tf.io.gfile.makedirs(eval_dir)
    self._model = model
    self._labels_file = labels_file
    self._params = params
    self._save_predictions = save_predictions
    self._scorers = scorers
    self._eval_dir = eval_dir
    self._dataset = model.examples_inputter.make_evaluation_dataset(
        features_file,
        labels_file,
        batch_size,
        num_threads=1,
        prefetch_buffer_size=1)

    @tf.function(input_signature=dataset_lib.input_signature_from_dataset(self._dataset))
    def _eval(source, target):
      outputs, predictions = model(source, target, params, tf.estimator.ModeKeys.EVAL)
      loss = model.compute_loss(outputs, target, training=False, params=params)
      return loss, predictions

    self._eval_function = _eval

  @classmethod
  def from_config(cls, model, config):
    """Creates an evaluator from the configuration.

    Args:
      model: A :class:`opennmt.models.model.Model` to evaluate.
      config: The global user configuration.

    Returns:
      A :class:`opennmt.evaluation.Evaluator` instance.
    """
    scorers = config["eval"].get("external_evaluators")
    if scorers is not None:
      scorers = make_scorers(scorers)
    return cls(
        model,
        config["data"]["eval_features_file"],
        config["data"].get("eval_labels_file"),
        config["eval"]["batch_size"],
        params=config.get("params"),
        scorers=scorers,
        save_predictions=config["eval"].get("save_eval_predictions", False),
        eval_dir=os.path.join(config["model_dir"], "eval"))

  def __call__(self, step):
    """Runs the evaluator.

    Args:
      step: The current training step.

    Returns:
      A dictionary of evaluation metrics.
    """
    tf.get_logger().info("Running evaluation for step %d", step)
    output_file = None
    output_path = None
    if self._save_predictions:
      output_path = os.path.join(self._eval_dir, "predictions.txt.%d" % step)
      output_file = open(output_path, "w")

    loss_num = 0
    loss_den = 0
    for source, target in self._dataset:
      loss, predictions = self._eval_function(source, target)
      loss_num += loss[0]
      loss_den += loss[1]
      if output_file is not None:
        predictions = {k:v.numpy() for k, v in six.iteritems(predictions)}
        for prediction in misc.extract_batches(predictions):
          self._model.print_prediction(prediction, stream=output_file)
    loss = loss_num / loss_den

    results = dict(loss=loss)
    if self._save_predictions:
      tf.get_logger().info("Evaluation predictions saved to %s", output_path)
      output_file.close()
      for scorer in self._scorers:
        score = scorer(self._labels_file, output_path)
        if isinstance(score, dict):
          results.update(score)
        else:
          results[scorer.name] = score

    tf.get_logger().info(
        "Evaluation result for step %d: %s",
        step,
        " ; ".join("%s = %f" % (k, v) for k, v in six.iteritems(results)))
    return results


@six.add_metaclass(abc.ABCMeta)
class Scorer(object):
  """Scores hypotheses against references."""

  def __init__(self, name):
    self._name = name

  @property
  def name(self):
    """The scorer name."""
    return self._name

  @abc.abstractmethod
  def __call__(self, ref_path, hyp_path):
    """Scores hypotheses.

    Args:
      ref_path: Path to the reference.
      hyp_path: Path to the hypotheses.

    Returns:
      The score or dictionary of scores.
    """
    raise NotImplementedError()

  def lower_is_better(self):
    """Returns ``True`` if a lower score is better."""
    return False


class ROUGEScorer(Scorer):
  """ROUGE scorer based on https://github.com/pltrdy/rouge."""

  def __init__(self):
    super(ROUGEScorer, self).__init__("rouge")

  def __call__(self, ref_path, hyp_path):
    from rouge import FilesRouge
    files_rouge = FilesRouge(hyp_path, ref_path)
    rouge_scores = files_rouge.get_scores(avg=True)
    return {k:v["f"] for k, v in six.iteritems(rouge_scores)}


class BLEUScorer(Scorer):
  """Scorer using sacreBLEU."""

  def __init__(self):
    try:
      import sacrebleu  # pylint: disable=unused-import, unused-variable
    except ImportError:
      raise ImportError("sacreBLEU evaluator requires Python 3")
    super(BLEUScorer, self).__init__("bleu")

  def __call__(self, ref_path, hyp_path):
    from sacrebleu import corpus_bleu
    with open(ref_path) as ref_stream, open(hyp_path) as sys_stream:
      bleu = corpus_bleu(sys_stream, [ref_stream])
      return bleu.score


def make_scorers(names):
  """Returns a list of scorers.

  Args:
    names: A list of scorer names or a single name.

  Returns:
    A list of :class:`opennmt.evaluation.Scorer` instances.

  Raises:
    ValueError: if a scorer name is invalid.
  """
  if not isinstance(names, list):
    names = [names]
  scorers = []
  for name in names:
    name = name.lower()
    scorer = None
    if name == "bleu":
      scorer = BLEUScorer()
    elif name == "rouge":
      scorer = ROUGEScorer()
    else:
      raise ValueError("No scorer associated with the name: {}".format(name))
    scorers.append(scorer)
  return scorers
