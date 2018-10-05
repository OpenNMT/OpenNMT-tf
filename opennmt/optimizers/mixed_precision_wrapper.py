# pylint: skip-file

# MIT License

# Copyright (c) 2018 NVIDIA Corporation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Wrapper that maintains and update a float32 copy of the weights."""

import six

import tensorflow as tf


class MixedPrecisionOptimizerWrapper(tf.train.Optimizer):
  def __init__(self, optimizer, loss_scale=None):
    super(MixedPrecisionOptimizerWrapper, self).__init__(
        optimizer._use_locking,
        optimizer._name + '-MP',
    )
    self._optimizer = optimizer
    self._fp32_to_fp16 = {}
    self._loss_scaler = None
    if loss_scale is None:
      self._loss_scale = 1.0
    elif isinstance(loss_scale, float):
      self._loss_scale = loss_scale
    elif isinstance(loss_scale, AutomaticLossScaler):
      self._loss_scaler = loss_scale
      self._loss_scale = self._loss_scaler.loss_scale

  def compute_gradients(self, loss, var_list=None,
                        gate_gradients=tf.train.Optimizer.GATE_OP,
                        aggregation_method=None,
                        colocate_gradients_with_ops=False,
                        grad_loss=None):
    loss *= self._loss_scale
    grads_and_vars_fp16 = self._optimizer.compute_gradients(
        loss, var_list=var_list,
        gate_gradients=gate_gradients,
        aggregation_method=aggregation_method,
        colocate_gradients_with_ops=colocate_gradients_with_ops,
        grad_loss=grad_loss,
    )

    # collecting regularization functions
    reg_var_funcs = tf.get_collection('REGULARIZATION_FUNCTIONS')
    reg_funcs = dict(map(lambda x: (x[0].name, x[1]), reg_var_funcs))

    # creating FP-32 variables and filling the fp32 dict
    grads_and_vars_fp32 = []
    with tf.variable_scope('FP32-master-copy'):
      for grad, var in grads_and_vars_fp16:
        if var.dtype.base_dtype == tf.float16:
          fp32_var = tf.Variable(
              initial_value=tf.cast(var.initialized_value(), tf.float32),
              name=var.name.split(':')[0],
              expected_shape=var.shape,
              dtype=tf.float32,
              trainable=False,
              # necessary for cudnn_rnn layers which have unknown shape
              validate_shape=bool(var.get_shape()),
              collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                           "FP32_MASTER_COPIES"],
          )
          self._fp32_to_fp16[fp32_var.name] = var
          fp32_grad = tf.cast(grad, tf.float32)
          # adding regularization part with respect to fp32 copy
          if var.name in reg_funcs:
            fp32_grad += self._loss_scale * tf.gradients(
                # pylint: disable=no-member
                tf.contrib.layers.apply_regularization(
                    reg_funcs[var.name],
                    [fp32_var],
                ),
                fp32_var,
            )[0]
          grads_and_vars_fp32.append((fp32_grad, fp32_var))
        else:
          grads_and_vars_fp32.append((grad, var))

    grads_and_vars_fp32 = _scale_grads(grads_and_vars_fp32,
                                       1.0 / self._loss_scale)
    return grads_and_vars_fp32

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    def apply_ops_wrapper():
      update_op = self._optimizer.apply_gradients(grads_and_vars,
                                                  global_step, name)
      apply_ops = []
      with tf.control_dependencies([update_op]):
        for grad, var in grads_and_vars:
          if var.name in self._fp32_to_fp16:
            dst_var = self._fp32_to_fp16[var.name]
            apply_ops.append(
                tf.assign(dst_var, tf.saturate_cast(var, tf.float16))
            )
      if apply_ops:
        return tf.group(apply_ops)
      return update_op

    if self._loss_scaler:
      grad_has_nans, grad_amax = AutomaticLossScaler.check_grads(grads_and_vars)
      should_skip_update = tf.logical_or(tf.is_inf(grad_amax), grad_has_nans)
      loss_scale_update_op = self._loss_scaler.update_op(grad_has_nans,
                                                         grad_amax)
      with tf.control_dependencies([loss_scale_update_op]):
        return tf.cond(should_skip_update, tf.no_op, apply_ops_wrapper)
    else:
      return apply_ops_wrapper()


def _scale_grads(grads_and_vars, scale):
  scaled_grads_and_vars = []
  for grad, var in grads_and_vars:
    if grad is not None:
      if isinstance(grad, tf.IndexedSlices):
        grad_values = grad.values * scale
        grad = tf.IndexedSlices(grad_values, grad.indices, grad.dense_shape)
      else:
        grad *= scale
    scaled_grads_and_vars.append((grad, var))
  return scaled_grads_and_vars


class AutomaticLossScaler(object):
  SUPPORTED_ALGOS = ['backoff', 'logmax']

  def __init__(self, algorithm='Backoff', params=None):
    algorithm = algorithm.lower().strip()
    if algorithm == 'backoff':
      self.scaler = BackoffScaler(params)
    elif algorithm == 'logmax':
      self.scaler = LogMaxScaler(params)  # ppf(.999)
    else:
      raise ValueError('Unknown scaling algorithm: {}'.format(algorithm))

  def update_op(self, has_nan, amax):
    return self.scaler.update_op(has_nan, amax)

  @property
  def loss_scale(self):
    return self.scaler.loss_scale

  @staticmethod
  def check_grads(grads_and_vars):
    has_nan_ops = []
    amax_ops = []

    for grad, _ in grads_and_vars:
      if grad is not None:
        if isinstance(grad, tf.IndexedSlices):
          x = grad.values
        else:
          x = grad

        has_nan_ops.append(tf.reduce_any(tf.is_nan(x)))
        amax_ops.append(tf.reduce_max(tf.abs(x)))

    has_nan = tf.reduce_any(has_nan_ops)
    amax = tf.reduce_max(amax_ops)
    return has_nan, amax


class BackoffScaler(object):
  def __init__(self, params):
    if params is None:
      params = {}
    self.scale_min = params.get('scale_min', 1.0)
    self.scale_max = params.get('scale_max', 2.**24)
    self.step_factor = params.get('step_factor', 2.0)
    self.step_window = params.get('step_window', 2000)

    self.iteration = tf.Variable(initial_value=0,
                                 trainable=False,
                                 dtype=tf.int64)
    self.last_overflow_iteration = tf.Variable(initial_value=-1,
                                               trainable=False,
                                               dtype=tf.int64)
    self.scale = tf.Variable(initial_value=self.scale_max,
                             trainable=False)

  def update_op(self, has_nan, amax):
    def overflow_case():
      new_scale_val = tf.clip_by_value(self.scale / self.step_factor,
                                       self.scale_min, self.scale_max)
      scale_assign = tf.assign(self.scale, new_scale_val)
      overflow_iter_assign = tf.assign(self.last_overflow_iteration,
                                       self.iteration)
      with tf.control_dependencies([scale_assign, overflow_iter_assign]):
        return tf.identity(self.scale)

    def scale_case():
      since_overflow = self.iteration - self.last_overflow_iteration
      should_update = tf.equal(since_overflow % self.step_window, 0)
      def scale_update_fn():
        new_scale_val = tf.clip_by_value(self.scale * self.step_factor,
                                         self.scale_min, self.scale_max)
        return tf.assign(self.scale, new_scale_val)
      return tf.cond(should_update,
                     scale_update_fn,
                     lambda: self.scale)

    iter_update = tf.assign_add(self.iteration, 1)
    overflow = tf.logical_or(has_nan, tf.is_inf(amax))

    update_op = tf.cond(overflow,
                        overflow_case,
                        scale_case)
    with tf.control_dependencies([update_op]):
      return tf.identity(iter_update)

  @property
  def loss_scale(self):
    return self.scale


class LogMaxScaler(object):
  def __init__(self, params):
    if params is None:
      params = {}
    self.scale_min = params.get('scale_min', 1.0)
    self.scale_max = params.get('scale_max', 2.**24)
    self.log_max = params.get('log_max', 16.)
    self.beta1 = params.get('beta1', 0.99)
    self.beta2 = params.get('beta2', 0.999)
    self.overflow_std_dev = params.get('overflow_std_dev', 3.09)

    self.iteration = tf.Variable(initial_value=0,
                                 trainable=False,
                                 dtype=tf.int64)
    self.scale = tf.Variable(initial_value=1.0,
                             trainable=False)
    self.x_hat = tf.Variable(initial_value=0,
                             trainable=False,
                             dtype=tf.float32)
    self.slow_x_hat = tf.Variable(initial_value=0,
                                  trainable=False,
                                  dtype=tf.float32)
    self.xsquared_hat = tf.Variable(initial_value=0,
                                    trainable=False,
                                    dtype=tf.float32)
    self.b1_correction = tf.Variable(initial_value=1.,
                                     trainable=False,
                                     dtype=tf.float32)
    self.b2_correction = tf.Variable(initial_value=1.,
                                     trainable=False,
                                     dtype=tf.float32)

  # NB: assumes that `amax` is already has been downscaled
  def update_op(self, has_nan, amax):
    is_nonfinite = tf.logical_or(has_nan, tf.is_inf(amax))
    x = tf.cond(is_nonfinite,
                lambda: tf.pow(2., self.log_max),
                lambda: tf.log(amax) / tf.log(tf.constant(2.)))

    x_hat_assn = tf.assign(self.x_hat, self.beta1 * self.x_hat +
                           (1 - self.beta1) * x)
    b1_corr_assn = tf.assign(self.b1_correction,
                             self.b1_correction * self.beta1)
    with tf.control_dependencies([x_hat_assn, b1_corr_assn]):
      mu = self.x_hat.read_value() / (1 - self.b1_correction.read_value())

    slow_x_hat_assn = tf.assign(self.slow_x_hat, self.beta2 * self.slow_x_hat +
                                (1 - self.beta2) * x)
    xsquared_hat_assn = tf.assign(
        self.xsquared_hat,
        self.beta2 * self.xsquared_hat + (1 - self.beta2) * (x * x),
    )
    b2_corr_assn = tf.assign(self.b2_correction,
                             self.b2_correction * self.beta2)
    with tf.control_dependencies([slow_x_hat_assn, xsquared_hat_assn,
                                  b2_corr_assn]):
      e_xsquared = self.xsquared_hat.read_value() / \
                   (1 - self.b2_correction.read_value())
      slow_mu = self.slow_x_hat.read_value() / \
                (1 - self.b2_correction.read_value())

    sigma2 = e_xsquared - (slow_mu * slow_mu)
    sigma = tf.sqrt(tf.maximum(sigma2, tf.constant(0.)))

    log_cutoff = sigma * self.overflow_std_dev + mu
    log_difference = 16 - log_cutoff
    proposed_scale = tf.pow(2., log_difference)
    scale_update = tf.assign(
        self.scale,
        tf.clip_by_value(proposed_scale, self.scale_min, self.scale_max),
    )
    iter_update = tf.assign_add(self.iteration, 1)

    with tf.control_dependencies([scale_update]):
      return tf.identity(iter_update)

  @property
  def loss_scale(self):
    return self.scale


# The following function(s) are not under the NVIDIA license:

def get_loss_scale_from_params(params):
  """Returns the loss scale argument from user parameters.

  Args:
    params: A dictionary containing the user parameters.

  Returns:
    A value that can be passed to the
     :class:`opennmt.optimizers.mixed_precision_wrapper.MixedPrecisionOptimizerWrapper`
    :obj:`loss_scale` constructor argument.
  """
  loss_scale = params.get("loss_scale")
  if loss_scale is None:
    return AutomaticLossScaler()
  if not isinstance(loss_scale, six.string_types):
    return loss_scale
  return AutomaticLossScaler(
      algorithm=loss_scale, params=params.get("loss_scale_params"))
