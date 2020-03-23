# -*- coding: utf-8 -*-

"""Competence Learning module."""

import tensorflow as tf
import math

class CurriculumLearner:
  """Defines curriculum learning filtering strategy."""

  def __init__(self, config):
    """Initializes the CurriculumLearner class.

    Args:
      config: training configuration file, curriculum learning is
        activated if "curriculum_learner" is present in `params` block.
    """
    self._difficulty_file = config["data"]["competence_learner"]
    if "competence_learner" in config["params"]:
      cfg_cl = config["params"]["competence_learner"]
      self._initial_value = cfg_cl.get("initial", 0.1)
      self._final_step = cfg_cl.get("final_step", 50000)
      if cfg_cl.get("type", "linear") == "linear":
        self._type = "linear"
      elif cfg_cl["type"] == "sqroot":
        self._type = "sqroot"
      else:
        raise ValueError("Invalid Competence Learner type")
      self._sample = cfg_cl.get("sample", False)
    else:
      self._type = "linear"
      self._final_step = 50000
      self._initial_value = 0.1
      self._sample = False
    self._competence = tf.Variable(1.0, dtype=tf.float32)
    self._count_filter_competence = tf.Variable((0,0), dtype=tf.int64)
    tf.get_logger().info("Initialize competence learner: %s, type: %s (%f/%d)",
               self._difficulty_file,
               self._type,
               self._initial_value,
               self._final_step)

  def config_has_curriculum_learning(config):
    """Checks if a given configuration activates Curriculum Learning.

    Args:
      config: the training configuration.
    """
    return "competence_learner" in config["data"]

  def set_rate(self, step):
    """Changes the rate of the competence learner - is called in the
      main training loop.

    Args:
      step: the current learning step.
    """
    if self._type == "linear":
      rate = step*(1.0-self._initial_value)/self._final_step+self._initial_value
    else:
      self._initial_value_square = self._initial_value*self._initial_value
      rate = math.sqrt(step*(1.0-self._initial_value_square)/self._final_step+self._initial_value_square)
    self._competence.assign(tf.cast(tf.math.minimum(rate, 1.0), tf.float32))
    return self

  def init_counter(self):
    """Initializes counter for reporting statistic on actual filtered
      sentences.
    """
    self._count_filter_competence.assign((0,0))
    return self

  @property
  def competence(self):
    """Returns the current competence.
    """
    return self._competence

  @property
  def filtered_rate(self):
    """Returns the current filtered rate since last call to `init_counter`.
  """
    if self._count_filter_competence[1]:
      return self._count_filter_competence.numpy()[0]*1./self._count_filter_competence[1].numpy()
    else:
      return 0.

  def filter(self):
    """Returns a dataset filter lambda function depending on current
      competence.
    """
    def _predicate(features_labels, difficulty):
      delta = difficulty-self._competence
      cond = delta <= 0 or (self._sample and tf.random.uniform(shape=[]) > delta) 
      self._count_filter_competence.assign_add((tf.cast(cond, dtype=tf.int64), 1))
      return tf.reduce_all(cond)

    return lambda dataset: dataset.filter(_predicate)
