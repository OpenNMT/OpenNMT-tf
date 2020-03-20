import tensorflow as tf
import math

class CompetenceLearner:

  def __init__(self, config):
    self._difficulty_file = config["data"]["competence_learner"]
    if "competence_learner" in config["params"]:
      cfg_cl = config["params"]["competence_learner"]
      self._initial_value = cfg_cl.get("initial", 0.01)
      self._final_step = cfg_cl.get("final_step", 50000)
      if cfg_cl.get("type", "linear") == "linear":
        self._type = "linear"
      elif cfg_cl["type"] == "sqroot":
        self._type = "sqroot"
      else:
        raise ValueError("Invalid Competence Learner type")
    else:
      self._type = "linear"
      self._final_step = 50000
      self._initial_value = 0.01
    self._competence = tf.Variable(1.0, dtype=tf.float32)
    self._count_filter_competence = tf.Variable((0,0), dtype=tf.int64)

  def set_rate(self, step):
    if self._type == "linear":
      rate = step*1.0/self._final_step+self._initial_value
    else:
      self._initial_value_square = self._initial_value*self._initial_value
      rate = math.sqrt(step*(1.0-self._initial_value_square)/self._final_step+self._initial_value_square)
    self._competence.assign(tf.cast(tf.math.minimum(rate, 1.0),tf.float32))
    return self

  def init_counter(self):
    self._count_filter_competence.assign((0,0))
    return self

  @property
  def filtered_rate(self):
    if self._count_filter_competence[1]:
      return self._count_filter_competence.numpy()[0]*1./self._count_filter_competence[1].numpy()
    else:
      return 0.

  def filter(self):
    def _predicate(features_labels, difficulty):
      cond = difficulty < self._competence
      self._count_filter_competence.assign_add((tf.cast(cond, dtype=tf.int64),1))
      return tf.reduce_all(cond)

    return lambda dataset: dataset.filter(_predicate)
