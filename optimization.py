# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""与优化（权重更新）相关的函数和类

该模块包含 BERT 模型训练中使用的优化器实现，包括：
- create_optimizer 函数：创建优化器训练操作
- AdamWeightDecayOptimizer 类：实现带有权重衰减的 Adam 优化器

这些组件用于 BERT 模型的预训练和微调过程中。
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf


def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps, use_tpu):
  """创建优化器训练操作

  实现学习率的线性衰减和预热，并使用 AdamWeightDecayOptimizer 进行权重更新。

  参数：
    loss: 损失张量
    init_lr: 初始学习率
    num_train_steps: 训练步数
    num_warmup_steps: 预热步数
    use_tpu: 是否使用 TPU

  返回：
    训练操作
  """

  # 获取或创建全局步数
  global_step = tf.train.get_or_create_global_step()

  # 初始化学习率为常数
  learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

  # 实现学习率的线性衰减
  learning_rate = tf.train.polynomial_decay(
      learning_rate,
      global_step,
      num_train_steps,
      end_learning_rate=0.0,
      power=1.0,  # 线性衰减
      cycle=False)

  # 实现学习率的线性预热
  # 如果 global_step < num_warmup_steps，学习率将是 `global_step/num_warmup_steps * init_lr`
  if num_warmup_steps:
    global_steps_int = tf.cast(global_step, tf.int32)
    warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

    global_steps_float = tf.cast(global_steps_int, tf.float32)
    warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

    warmup_percent_done = global_steps_float / warmup_steps_float
    warmup_learning_rate = init_lr * warmup_percent_done

    is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
    learning_rate = (
        (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

  # 建议使用此优化器进行微调，因为模型就是这样训练的
  # 注意：Adam 的 m/v 变量不会从 init_checkpoint 加载
  optimizer = AdamWeightDecayOptimizer(
      learning_rate=learning_rate,
      weight_decay_rate=0.01,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-6,
      exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

  # 如果使用 TPU，包装优化器
  if use_tpu:
    optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

  # 获取所有可训练变量
  tvars = tf.trainable_variables()
  # 计算梯度
  grads = tf.gradients(loss, tvars)

  # 梯度裁剪，这是模型预训练时使用的方式
  (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

  # 应用梯度更新
  train_op = optimizer.apply_gradients(
      zip(grads, tvars), global_step=global_step)

  # 通常全局步数更新是在 `apply_gradients` 内部完成的
  # 然而，`AdamWeightDecayOptimizer` 不会这样做
  # 但如果你使用不同的优化器，你可能应该删除这行
  new_global_step = global_step + 1
  train_op = tf.group(train_op, [global_step.assign(new_global_step)])
  return train_op



class AdamWeightDecayOptimizer(tf.train.Optimizer):
  """带有"正确" L2 权重衰减的基本 Adam 优化器

  该优化器实现了 Adam 优化算法，并添加了权重衰减功能，
  但与标准 Adam 不同，它以不与动量参数相互作用的方式应用权重衰减。
  """


  def __init__(self,
               learning_rate,
               weight_decay_rate=0.0,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               exclude_from_weight_decay=None,
               name="AdamWeightDecayOptimizer"):
    """构造 AdamWeightDecayOptimizer

    参数：
      learning_rate: 学习率
      weight_decay_rate: 权重衰减率
      beta_1: Adam 的 beta1 参数
      beta_2: Adam 的 beta2 参数
      epsilon: Adam 的 epsilon 参数
      exclude_from_weight_decay: 要排除在权重衰减之外的变量名列表
      name: 优化器名称
    """
    super(AdamWeightDecayOptimizer, self).__init__(False, name)

    self.learning_rate = learning_rate
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.exclude_from_weight_decay = exclude_from_weight_decay


  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """应用梯度更新

    参数：
      grads_and_vars: 梯度和变量的列表
      global_step: 全局步数
      name: 操作名称

    返回：
      应用梯度更新的操作
    """

    assignments = []
    for (grad, param) in grads_and_vars:
      if grad is None or param is None:
        continue

      # 获取参数名称
      param_name = self._get_variable_name(param.name)

      # 创建动量变量 m
      m = tf.get_variable(
          name=param_name + "/adam_m",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())
      # 创建速度变量 v
      v = tf.get_variable(
          name=param_name + "/adam_v",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())

      # 标准 Adam 更新
      next_m = (
          tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
      next_v = (
          tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                    tf.square(grad)))

      # 计算更新值
      update = next_m / (tf.sqrt(next_v) + self.epsilon)

      # 注意：在损失函数中简单添加权重的平方并不是使用 L2 正则化/权重衰减的正确方法，
      # 因为这会以奇怪的方式与 m 和 v 参数相互作用。
      # 相反，我们希望以不与 m/v 参数相互作用的方式衰减权重。
      # 这相当于在普通（非动量）SGD 中向损失添加权重的平方。
      if self._do_use_weight_decay(param_name):
        update += self.weight_decay_rate * param

      # 应用学习率
      update_with_lr = self.learning_rate * update

      # 计算新的参数值
      next_param = param - update_with_lr

      # 添加参数更新和动量变量更新
      assignments.extend(
          [param.assign(next_param),
           m.assign(next_m),
           v.assign(next_v)])
    # 组合所有更新操作
    return tf.group(*assignments, name=name)


  def _do_use_weight_decay(self, param_name):
    """是否对 `param_name` 使用 L2 权重衰减

    参数：
      param_name: 参数名称

    返回：
      bool. 如果应该对该参数使用权重衰减，则为 True
    """

    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _get_variable_name(self, param_name):
    """从张量名称获取变量名称

    参数：
      param_name: 张量名称

    返回：
      变量名称
    """

    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name
