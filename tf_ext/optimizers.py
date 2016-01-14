"""NormOptimiser for TensorFlow."""

from tensorflow.python.framework import ops
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops

import tensorflow as tf


class AdamPlusOptimizer(optimizer.Optimizer):
    """Optimizer that implements the Adam algorithm.

    @@__init__
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, pow=0.7,
                 use_locking=False, name="Norm"):
        """Construct a modified Adam optimizer.

        Implementation is based on: http://arxiv.org/pdf/1412.6980v7.pdf

        Initialization:

        ```
        m_0 <- 0 (Initialize initial 1st moment vector)
        v_0 <- 0 (Initialize initial 2nd moment vector)
        t <- 0 (Initialize timestep)
        ```

        The update rule for `variable` with gradient `g` uses an optimization
        described at the end of section2 of the paper:

        ```
        t <- t + 1
        lr_t <- learning_rate * sqrt(1 - beta2^t) / (1 - beta1^t)

        m_t <- beta1 * m_{t-1} + (1 - beta1) * g
        v_t <- beta2 * v_{t-1} + (1 - beta2) * g * g
        variable <- variable - lr_t * m_t / (sqrt(v_t) + epsilon)
        ```

        The default value of 1e-8 for epsilon might not be a good default in
        general. For example, when training an Inception network on ImageNet a
        current good choice is 1.0 or 0.1.

        Args:
          learning_rate: A Tensor or a floating point value.  The learning rate.
          beta1: A float value or a constant float tensor.
            The exponential decay rate for the 1st moment estimates.
          beta2: A float value or a constant float tensor.
            The exponential decay rate for the 2nd moment estimates.
          epsilon: A small constant for numerical stability.
          use_locking: If True use locks for update operations.
          name: Optional name for the operations created when applying gradients.
            Defaults to "Adam".
        """
        super(AdamPlusOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._pow = pow

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None
        self._pow_t = None

        # Variables to accumulate the powers of the beta parameters.
        # Created in _create_slots when we know the variables to optimize.
        self._beta1_power = None
        self._beta2_power = None

        # Created in SparseApply if needed.
        self._updated_lr = None

    def _get_beta_accumulators(self):
        return self._beta1_power, self._beta2_power

    def _create_slots(self, var_list):
        # Create the beta1 and beta2 accumulators on the same device as the first
        # variable.
        if self._beta1_power is None:
            with ops.device(var_list[0].device):
                self._beta1_power = variables.Variable(self._beta1,
                                                       name="beta1_power",
                                                       trainable=False)
                self._beta2_power = variables.Variable(self._beta2,
                                                       name="beta2_power",
                                                       trainable=False)
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")
        self._epsilon_t = ops.convert_to_tensor(self._epsilon, name="epsilon")
        self._pow_t = ops.convert_to_tensor(self._pow, name="pow")

    def _apply_dense(self, grad, var):
        lr = (self._lr_t *
              math_ops.sqrt(1 - self._beta2_power)
              / (1 - self._beta1_power))
        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * (1 - self._beta1_t)
        m_t = m * self._beta1_t
        m_t = m_t + m_scaled_g_values
        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_scaled_g_values = tf.pow(grad, 2) * (1 - self._beta2_t)
        v_t = v * self._beta2_t
        v_t = v_t + v_scaled_g_values
        v_sqrt = tf.pow(v_t, self._pow_t)
        var_update = state_ops.assign_sub(var,
                                          lr * m_t / (v_sqrt + self._epsilon_t),
                                          use_locking=self._use_locking)

        return control_flow_ops.group(*[var_update, m_t, v_t])

    def _apply_sparse(self, grad, var):
        lr = (self._lr_t *
              math_ops.sqrt(1 - self._beta2_power)
              / (1 - self._beta1_power))
        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = grad.values * (1 - self._beta1_t)
        m_t = state_ops.assign(m, m * self._beta1_t,
                               use_locking=self._use_locking)

        m_t = state_ops.scatter_add(m_t, grad.indices, m_scaled_g_values,
                                    use_locking=self._use_locking)
        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_scaled_g_values = (grad.values * grad.values) * (1 - self._beta2_t)
        v_t = state_ops.assign(v, v * self._beta2_t, use_locking=self._use_locking)
        v_t = state_ops.scatter_add(v_t, grad.indices, v_scaled_g_values,
                                    use_locking=self._use_locking)
        v_sqrt = tf.pow(v_t, self._pow_t)
        var_update = state_ops.assign_sub(var,
                                          lr * m_t / (v_sqrt + self._epsilon_t),
                                          use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, m_t, v_t])

    def _finish(self, update_ops, name_scope):
        # Update the power accumulators.
        with ops.control_dependencies(update_ops):
            with ops.device(self._beta1_power.device):
                update_beta1 = self._beta1_power.assign(
                        self._beta1_power * self._beta1_t,
                        use_locking=self._use_locking)
                update_beta2 = self._beta2_power.assign(
                        self._beta2_power * self._beta2_t,
                        use_locking=self._use_locking)
        return control_flow_ops.group(*update_ops + [update_beta1, update_beta2],
                                      name=name_scope)


class AdamPlusCovOptimizer(optimizer.Optimizer):
    """Optimizer that implements the Adam algorithm.

    @@__init__
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, pow=0.7,
                 use_locking=False, name="Norm"):
        """Construct a modified Adam optimizer.

        Implementation is based on: http://arxiv.org/pdf/1412.6980v7.pdf

        Initialization:

        ```
        m_0 <- 0 (Initialize initial 1st moment vector)
        v_0 <- 0 (Initialize initial 2nd moment vector)
        t <- 0 (Initialize timestep)
        ```

        The update rule for `variable` with gradient `g` uses an optimization
        described at the end of section2 of the paper:

        ```
        t <- t + 1
        lr_t <- learning_rate * sqrt(1 - beta2^t) / (1 - beta1^t)

        m_t <- beta1 * m_{t-1} + (1 - beta1) * g
        v_t <- beta2 * v_{t-1} + (1 - beta2) * g * g
        variable <- variable - lr_t * m_t / (sqrt(v_t) + epsilon)
        ```

        The default value of 1e-8 for epsilon might not be a good default in
        general. For example, when training an Inception network on ImageNet a
        current good choice is 1.0 or 0.1.

        Args:
          learning_rate: A Tensor or a floating point value.  The learning rate.
          beta1: A float value or a constant float tensor.
            The exponential decay rate for the 1st moment estimates.
          beta2: A float value or a constant float tensor.
            The exponential decay rate for the 2nd moment estimates.
          epsilon: A small constant for numerical stability.
          use_locking: If True use locks for update operations.
          name: Optional name for the operations created when applying gradients.
            Defaults to "Adam".
        """
        super(AdamPlusCovOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._pow = pow

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None
        self._pow_t = None

        # Variables to accumulate the powers of the beta parameters.
        # Created in _create_slots when we know the variables to optimize.
        self._beta1_power = None
        self._beta2_power = None

        # Created in SparseApply if needed.
        self._updated_lr = None

    def _get_beta_accumulators(self):
        return self._beta1_power, self._beta2_power

    def _create_slots(self, var_list):
        # Create the beta1 and beta2 accumulators on the same device as the first
        # variable.
        if self._beta1_power is None:
            with ops.device(var_list[0].device):
                self._beta1_power = variables.Variable(self._beta1,
                                                       name="beta1_power",
                                                       trainable=False)
                self._beta2_power = variables.Variable(self._beta2,
                                                       name="beta2_power",
                                                       trainable=False)
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")
        self._epsilon_t = ops.convert_to_tensor(self._epsilon, name="epsilon")
        self._pow_t = ops.convert_to_tensor(self._pow, name="pow")

    def _apply_dense(self, grad, var):
        lr = (self._lr_t *
              math_ops.sqrt(1 - self._beta2_power)
              / (1 - self._beta1_power))
        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * (1 - self._beta1_t)
        m_t = m * self._beta1_t
        m_t = m_t + m_scaled_g_values
        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        #######################################################################################
        v_scaled_g_values = tf.pow(grad - m_t, 2) * (1 - self._beta2_t)
        #######################################################################################
        v_t = v * self._beta2_t
        v_t = v_t + v_scaled_g_values
        v_sqrt = tf.pow(v_t, self._pow_t)
        var_update = state_ops.assign_sub(var,
                                          lr * m_t / (v_sqrt + self._epsilon_t),
                                          use_locking=self._use_locking)

        return control_flow_ops.group(*[var_update, m_t, v_t])

    def _apply_sparse(self, grad, var):
        lr = (self._lr_t *
              math_ops.sqrt(1 - self._beta2_power)
              / (1 - self._beta1_power))
        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = grad.values * (1 - self._beta1_t)
        m_t = state_ops.assign(m, m * self._beta1_t,
                               use_locking=self._use_locking)

        m_t = state_ops.scatter_add(m_t, grad.indices, m_scaled_g_values,
                                    use_locking=self._use_locking)
        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_scaled_g_values = (grad.values * grad.values) * (1 - self._beta2_t)
        v_t = state_ops.assign(v, v * self._beta2_t, use_locking=self._use_locking)
        v_t = state_ops.scatter_add(v_t, grad.indices, v_scaled_g_values,
                                    use_locking=self._use_locking)
        v_sqrt = tf.pow(v_t, self._pow_t)
        var_update = state_ops.assign_sub(var,
                                          lr * m_t / (v_sqrt + self._epsilon_t),
                                          use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, m_t, v_t])

    def _finish(self, update_ops, name_scope):
        # Update the power accumulators.
        with ops.control_dependencies(update_ops):
            with ops.device(self._beta1_power.device):
                update_beta1 = self._beta1_power.assign(
                        self._beta1_power * self._beta1_t,
                        use_locking=self._use_locking)
                update_beta2 = self._beta2_power.assign(
                        self._beta2_power * self._beta2_t,
                        use_locking=self._use_locking)
        return control_flow_ops.group(*update_ops + [update_beta1, update_beta2],
                                      name=name_scope)
