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
"""BERT 模型的核心实现和相关函数

该模块包含 BERT（Bidirectional Encoder Representations from Transformers）模型的主要实现，
包括模型配置、模型架构、注意力机制、嵌入层等核心组件。

主要功能：
- BertConfig 类：管理 BERT 模型的配置参数
- BertModel 类：实现 BERT 模型的完整架构
- 各种辅助函数：处理嵌入、注意力计算、层归一化等

使用示例：
```python
# 已经转换为 WordPiece 标记 ID
input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
  num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

model = modeling.BertModel(config=config, is_training=True,
  input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

label_embeddings = tf.get_variable(...)
pooled_output = model.get_pooled_output()
logits = tf.matmul(pooled_output, label_embeddings)
...
```
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import json
import math
import re
import numpy as np
import six
import tensorflow as tf


class BertConfig(object):
  """BERT 模型的配置类

  管理 BERT 模型的所有配置参数，包括模型架构、大小、dropout 率等。
  """

  def __init__(self,
               vocab_size,
               hidden_size=768,
               num_hidden_layers=12,
               num_attention_heads=12,
               intermediate_size=3072,
               hidden_act="gelu",
               hidden_dropout_prob=0.1,
               attention_probs_dropout_prob=0.1,
               max_position_embeddings=512,
               type_vocab_size=16,
               initializer_range=0.02):
    """构造 BertConfig 实例

    参数：
      vocab_size: BERT 模型中 `input_ids` 的词汇表大小
      hidden_size: 编码器层和池化层的大小
      num_hidden_layers: Transformer 编码器中的隐藏层数量
      num_attention_heads: Transformer 编码器中每个注意力层的注意力头数
      intermediate_size: Transformer 编码器中"中间"（即前馈）层的大小
      hidden_act: 编码器和池化器中的非线性激活函数（函数或字符串）
      hidden_dropout_prob: 嵌入、编码器和池化器中所有全连接层的 dropout 概率
      attention_probs_dropout_prob: 注意力概率的 dropout 比率
      max_position_embeddings: 此模型可能使用的最大序列长度
      type_vocab_size: 传递给 `BertModel` 的 `token_type_ids` 的词汇表大小
      initializer_range: 用于初始化所有权重矩阵的 truncated_normal_initializer 的标准差
    """
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.hidden_act = hidden_act
    self.intermediate_size = intermediate_size
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.max_position_embeddings = max_position_embeddings
    self.type_vocab_size = type_vocab_size
    self.initializer_range = initializer_range

  @classmethod
  def from_dict(cls, json_object):
    """从 Python 参数字典构造 `BertConfig`

    参数：
      json_object: 包含配置参数的字典

    返回：
      BertConfig 实例
    """
    config = BertConfig(vocab_size=None)
    for (key, value) in six.iteritems(json_object):
      config.__dict__[key] = value
    return config

  @classmethod
  def from_json_file(cls, json_file):
    """从 JSON 文件构造 `BertConfig`

    参数：
      json_file: 包含配置参数的 JSON 文件路径

    返回：
      BertConfig 实例
    """
    with tf.gfile.GFile(json_file, "r") as reader:
      text = reader.read()
    return cls.from_dict(json.loads(text))

  def to_dict(self):
    """将此实例序列化为 Python 字典

    返回：
      包含所有配置参数的字典
    """
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    """将此实例序列化为 JSON 字符串

    返回：
      包含所有配置参数的 JSON 字符串
    """
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"



class BertModel(object):
  """BERT 模型（"Bidirectional Encoder Representations from Transformers"）

  BERT 是一种预训练语言模型，通过双向Transformer编码器捕获文本的上下文信息。

  使用示例：

  ```python
  # 已经转换为 WordPiece 标记 ID
  input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
  input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
  token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

  config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
    num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

  model = modeling.BertModel(config=config, is_training=True,
    input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

  label_embeddings = tf.get_variable(...)
  pooled_output = model.get_pooled_output()
  logits = tf.matmul(pooled_output, label_embeddings)
  ...
  ```
  """

  def __init__(self,
               config,
               is_training,
               input_ids,
               input_mask=None,
               token_type_ids=None,
               use_one_hot_embeddings=False,
               scope=None):
    """BertModel 构造函数

    参数：
      config: `BertConfig` 实例
      is_training: bool. 训练模型为 true，评估模型为 false。控制是否应用 dropout
      input_ids: 形状为 [batch_size, seq_length] 的 int32 张量
      input_mask: (可选) 形状为 [batch_size, seq_length] 的 int32 张量
      token_type_ids: (可选) 形状为 [batch_size, seq_length] 的 int32 张量
      use_one_hot_embeddings: (可选) bool. 是否使用 one-hot 词嵌入或 tf.embedding_lookup()
      scope: (可选) 变量作用域。默认为 "bert"

    异常：
      ValueError: 配置无效或输入张量形状无效
    """

    config = copy.deepcopy(config)
    if not is_training:
      config.hidden_dropout_prob = 0.0
      config.attention_probs_dropout_prob = 0.0

    input_shape = get_shape_list(input_ids, expected_rank=2)
    batch_size = input_shape[0]
    seq_length = input_shape[1]

    if input_mask is None:
      input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

    if token_type_ids is None:
      token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

    with tf.variable_scope(scope, default_name="bert"):
      with tf.variable_scope("embeddings"):
        # Perform embedding lookup on the word ids.
        (self.embedding_output, self.embedding_table) = embedding_lookup(
            input_ids=input_ids,
            vocab_size=config.vocab_size,
            embedding_size=config.hidden_size,
            initializer_range=config.initializer_range,
            word_embedding_name="word_embeddings",
            use_one_hot_embeddings=use_one_hot_embeddings)

        # Add positional embeddings and token type embeddings, then layer
        # normalize and perform dropout.
        self.embedding_output = embedding_postprocessor(
            input_tensor=self.embedding_output,
            use_token_type=True,
            token_type_ids=token_type_ids,
            token_type_vocab_size=config.type_vocab_size,
            token_type_embedding_name="token_type_embeddings",
            use_position_embeddings=True,
            position_embedding_name="position_embeddings",
            initializer_range=config.initializer_range,
            max_position_embeddings=config.max_position_embeddings,
            dropout_prob=config.hidden_dropout_prob)

      with tf.variable_scope("encoder"):
        # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
        # mask of shape [batch_size, seq_length, seq_length] which is used
        # for the attention scores.
        attention_mask = create_attention_mask_from_input_mask(
            input_ids, input_mask)

        # Run the stacked transformer.
        # `sequence_output` shape = [batch_size, seq_length, hidden_size].
        self.all_encoder_layers = transformer_model(
            input_tensor=self.embedding_output,
            attention_mask=attention_mask,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            intermediate_act_fn=get_activation(config.hidden_act),
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            initializer_range=config.initializer_range,
            do_return_all_layers=True)

      self.sequence_output = self.all_encoder_layers[-1]
      # The "pooler" converts the encoded sequence tensor of shape
      # [batch_size, seq_length, hidden_size] to a tensor of shape
      # [batch_size, hidden_size]. This is necessary for segment-level
      # (or segment-pair-level) classification tasks where we need a fixed
      # dimensional representation of the segment.
      with tf.variable_scope("pooler"):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token. We assume that this has been pre-trained
        first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
        self.pooled_output = tf.layers.dense(
            first_token_tensor,
            config.hidden_size,
            activation=tf.tanh,
            kernel_initializer=create_initializer(config.initializer_range))

  def get_pooled_output(self):
    """获取池化层输出

    返回：
      形状为 [batch_size, hidden_size] 的 float 张量，对应于整个序列的表示
    """
    return self.pooled_output

  def get_sequence_output(self):
    """获取编码器的最终隐藏层

    返回：
      形状为 [batch_size, seq_length, hidden_size] 的 float 张量，
      对应于 transformer 编码器的最终隐藏状态
    """
    return self.sequence_output

  def get_all_encoder_layers(self):
    """获取所有编码器层的输出

    返回：
      编码器各层输出的列表，每层都是形状为 [batch_size, seq_length, hidden_size] 的 float 张量
    """
    return self.all_encoder_layers

  def get_embedding_output(self):
    """获取嵌入查找的输出（即 transformer 的输入）

    返回：
      形状为 [batch_size, seq_length, hidden_size] 的 float 张量，
      对应于嵌入层的输出，在将词嵌入与位置嵌入和标记类型嵌入相加后，
      然后执行层归一化。这是 transformer 的输入
    """
    return self.embedding_output

  def get_embedding_table(self):
    """获取嵌入表

    返回：
      形状为 [vocab_size, hidden_size] 的嵌入表张量
    """
    return self.embedding_table



def gelu(x):
  """高斯误差线性单元（Gaussian Error Linear Unit）

  这是 RELU 的平滑版本，在 BERT 中用作激活函数。
  原始论文：https://arxiv.org/abs/1606.08415
  
  参数：
    x: 要执行激活的 float 张量

  返回：
    应用了 GELU 激活的 `x`
  """

  cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf


def get_activation(activation_string):
  """将字符串映射到 Python 函数，例如 "relu" => `tf.nn.relu`

  参数：
    activation_string: 激活函数的字符串名称

  返回：
    对应于激活函数的 Python 函数。如果 `activation_string` 为 None、空或 "linear"，则返回 None。
    如果 `activation_string` 不是字符串，则返回 `activation_string` 本身。

  异常：
    ValueError: `activation_string` 不对应于已知的激活函数
  """


  # We assume that anything that"s not a string is already an activation
  # function, so we just return it.
  if not isinstance(activation_string, six.string_types):
    return activation_string

  if not activation_string:
    return None

  act = activation_string.lower()
  if act == "linear":
    return None
  elif act == "relu":
    return tf.nn.relu
  elif act == "gelu":
    return gelu
  elif act == "tanh":
    return tf.tanh
  else:
    raise ValueError("Unsupported activation: %s" % act)


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
  """计算当前变量和检查点变量的并集

  构建从检查点变量名称到当前变量的映射，用于加载预训练模型。

  参数：
    tvars: 当前模型的变量列表
    init_checkpoint: 预训练模型检查点的路径

  返回：
    一个元组 (assignment_map, initialized_variable_names)，其中：
    - assignment_map: 从检查点变量名称到当前变量名称的映射
    - initialized_variable_names: 已初始化变量的名称集合
  """

  assignment_map = {}
  initialized_variable_names = {}

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var

  init_vars = tf.train.list_variables(init_checkpoint)

  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (name, var) = (x[0], x[1])
    if name not in name_to_variable:
      continue
    assignment_map[name] = name
    initialized_variable_names[name] = 1
    initialized_variable_names[name + ":0"] = 1

  return (assignment_map, initialized_variable_names)


def dropout(input_tensor, dropout_prob):
  """执行 dropout 操作

  在训练过程中随机丢弃部分神经元，以防止过拟合。

  参数：
    input_tensor: float 张量
    dropout_prob: Python float. 丢弃值的概率（不是 `tf.nn.dropout` 中的保留维度概率）

  返回：
    应用了 dropout 的 `input_tensor` 版本
  """

  if dropout_prob is None or dropout_prob == 0.0:
    return input_tensor

  output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
  return output


def layer_norm(input_tensor, name=None):
  """对张量的最后一个维度执行层归一化

  层归一化有助于稳定模型训练，加速收敛。

  参数：
    input_tensor: 输入张量
    name: (可选) 操作的名称

  返回：
    归一化后的张量
  """

  return tf.contrib.layers.layer_norm(
      inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


def layer_norm_and_dropout(input_tensor, dropout_prob, name=None):
  """执行层归一化后紧跟 dropout 操作

  这是 BERT 模型中常用的组合操作，用于处理层输出。

  参数：
    input_tensor: 输入张量
    dropout_prob: dropout 概率
    name: (可选) 操作的名称

  返回：
    经过层归一化和 dropout 处理的张量
  """

  output_tensor = layer_norm(input_tensor, name)
  output_tensor = dropout(output_tensor, dropout_prob)
  return output_tensor


def create_initializer(initializer_range=0.02):
  """创建具有给定范围的 `truncated_normal_initializer`

  用于初始化 BERT 模型中的权重矩阵。

  参数：
    initializer_range: 初始化器的标准差范围，默认为 0.02

  返回：
    truncated_normal_initializer 实例
  """

  return tf.truncated_normal_initializer(stddev=initializer_range)


def embedding_lookup(input_ids,
                     vocab_size,
                     embedding_size=128,
                     initializer_range=0.02,
                     word_embedding_name="word_embeddings",
                     use_one_hot_embeddings=False):
  """查找词嵌入

  根据输入的词 ID 张量查找对应的词嵌入。

  参数：
    input_ids: 形状为 [batch_size, seq_length] 的 int32 张量，包含词 ID
    vocab_size: int. 嵌入词汇表的大小
    embedding_size: int. 词嵌入的宽度
    initializer_range: float. 嵌入初始化范围
    word_embedding_name: string. 嵌入表的名称
    use_one_hot_embeddings: bool. 如果为 True，使用 one-hot 方法获取词嵌入；如果为 False，使用 `tf.gather()`

  返回：
    形状为 [batch_size, seq_length, embedding_size] 的 float 张量
  """

  # This function assumes that the input is of shape [batch_size, seq_length,
  # num_inputs].
  #
  # If the input is a 2D tensor of shape [batch_size, seq_length], we
  # reshape to [batch_size, seq_length, 1].
  if input_ids.shape.ndims == 2:
    input_ids = tf.expand_dims(input_ids, axis=[-1])

  embedding_table = tf.get_variable(
      name=word_embedding_name,
      shape=[vocab_size, embedding_size],
      initializer=create_initializer(initializer_range))

  flat_input_ids = tf.reshape(input_ids, [-1])
  if use_one_hot_embeddings:
    one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
    output = tf.matmul(one_hot_input_ids, embedding_table)
  else:
    output = tf.gather(embedding_table, flat_input_ids)

  input_shape = get_shape_list(input_ids)

  output = tf.reshape(output,
                      input_shape[0:-1] + [input_shape[-1] * embedding_size])
  return (output, embedding_table)


def embedding_postprocessor(input_tensor,
                            use_token_type=False,
                            token_type_ids=None,
                            token_type_vocab_size=16,
                            token_type_embedding_name="token_type_embeddings",
                            use_position_embeddings=True,
                            position_embedding_name="position_embeddings",
                            initializer_range=0.02,
                            max_position_embeddings=512,
                            dropout_prob=0.1):
  """对词嵌入张量执行各种后处理操作

  添加标记类型嵌入和位置嵌入，然后执行层归一化和 dropout。

  参数：
    input_tensor: 形状为 [batch_size, seq_length, embedding_size] 的 float 张量
    use_token_type: bool. 是否添加 `token_type_ids` 的嵌入
    token_type_ids: (可选) 形状为 [batch_size, seq_length] 的 int32 张量。如果 `use_token_type` 为 True，则必须指定
    token_type_vocab_size: int. `token_type_ids` 的词汇表大小
    token_type_embedding_name: string. 标记类型 ID 的嵌入表变量名称
    use_position_embeddings: bool. 是否添加序列中每个标记位置的位置嵌入
    position_embedding_name: string. 位置嵌入的嵌入表变量名称
    initializer_range: float. 权重初始化范围
    max_position_embeddings: int. 此模型可能使用的最大序列长度
    dropout_prob: float. 应用于最终输出张量的 dropout 概率

  返回：
    与 `input_tensor` 形状相同的 float 张量

  异常：
    ValueError: 张量形状或输入值无效
  """

  input_shape = get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  width = input_shape[2]

  output = input_tensor

  if use_token_type:
    if token_type_ids is None:
      raise ValueError("`token_type_ids` must be specified if"
                       "`use_token_type` is True.")
    token_type_table = tf.get_variable(
        name=token_type_embedding_name,
        shape=[token_type_vocab_size, width],
        initializer=create_initializer(initializer_range))
    # This vocab will be small so we always do one-hot here, since it is always
    # faster for a small vocabulary.
    flat_token_type_ids = tf.reshape(token_type_ids, [-1])
    one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)
    token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
    token_type_embeddings = tf.reshape(token_type_embeddings,
                                       [batch_size, seq_length, width])
    output += token_type_embeddings

  if use_position_embeddings:
    assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
    with tf.control_dependencies([assert_op]):
      full_position_embeddings = tf.get_variable(
          name=position_embedding_name,
          shape=[max_position_embeddings, width],
          initializer=create_initializer(initializer_range))
      # Since the position embedding table is a learned variable, we create it
      # using a (long) sequence length `max_position_embeddings`. The actual
      # sequence length might be shorter than this, for faster training of
      # tasks that do not have long sequences.
      #
      # So `full_position_embeddings` is effectively an embedding table
      # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
      # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
      # perform a slice.
      position_embeddings = tf.slice(full_position_embeddings, [0, 0],
                                     [seq_length, -1])
      num_dims = len(output.shape.as_list())

      # Only the last two dimensions are relevant (`seq_length` and `width`), so
      # we broadcast among the first dimensions, which is typically just
      # the batch size.
      position_broadcast_shape = []
      for _ in range(num_dims - 2):
        position_broadcast_shape.append(1)
      position_broadcast_shape.extend([seq_length, width])
      position_embeddings = tf.reshape(position_embeddings,
                                       position_broadcast_shape)
      output += position_embeddings

  output = layer_norm_and_dropout(output, dropout_prob)
  return output


def create_attention_mask_from_input_mask(from_tensor, to_mask):
  """从 2D 张量掩码创建 3D 注意力掩码

  用于在注意力机制中屏蔽填充位置。

  参数：
    from_tensor: 形状为 [batch_size, from_seq_length, ...] 的 2D 或 3D 张量
    to_mask: 形状为 [batch_size, to_seq_length] 的 int32 张量

  返回：
    形状为 [batch_size, from_seq_length, to_seq_length] 的 float 张量
  """

  from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
  batch_size = from_shape[0]
  from_seq_length = from_shape[1]

  to_shape = get_shape_list(to_mask, expected_rank=2)
  to_seq_length = to_shape[1]

  to_mask = tf.cast(
      tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

  # We don't assume that `from_tensor` is a mask (although it could be). We
  # don't actually care if we attend *from* padding tokens (only *to* padding)
  # tokens so we create a tensor of all ones.
  #
  # `broadcast_ones` = [batch_size, from_seq_length, 1]
  broadcast_ones = tf.ones(
      shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

  # Here we broadcast along two dimensions to create the mask.
  mask = broadcast_ones * to_mask

  return mask


def attention_layer(from_tensor,
                    to_tensor,
                    attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None):
  """执行从 `from_tensor` 到 `to_tensor` 的多头注意力

  这是基于 "Attention is all you Need" 论文的多头注意力实现。如果 `from_tensor` 和 `to_tensor` 相同，则为自注意力。
  `from_tensor` 中的每个时间步都会关注 `to_tensor` 中的对应序列，并返回一个固定宽度的向量。

  此函数首先将 `from_tensor` 投影到 "query" 张量，将 `to_tensor` 投影到 "key" 和 "value" 张量。
  这些实际上是长度为 `num_attention_heads` 的张量列表，每个张量的形状为 [batch_size, seq_length, size_per_head]。

  然后，query 和 key 张量进行点积并缩放，经过 softmax 得到注意力概率。然后用这些概率对 value 张量进行插值，
  最后将结果连接回单个张量并返回。

  在实践中，多头注意力是通过转置和重塑实现的，而不是实际的分离张量。

  参数：
    from_tensor: 形状为 [batch_size, from_seq_length, from_width] 的 float 张量
    to_tensor: 形状为 [batch_size, to_seq_length, to_width] 的 float 张量
    attention_mask: (可选) 形状为 [batch_size, from_seq_length, to_seq_length] 的 int32 张量，值应为 1 或 0
    num_attention_heads: int. 注意力头的数量
    size_per_head: int. 每个注意力头的大小
    query_act: (可选) query 变换的激活函数
    key_act: (可选) key 变换的激活函数
    value_act: (可选) value 变换的激活函数
    attention_probs_dropout_prob: (可选) float. 注意力概率的 dropout 概率
    initializer_range: float. 权重初始化的范围
    do_return_2d_tensor: bool. 如果为 True，输出形状为 [batch_size * from_seq_length, num_attention_heads * size_per_head]
    batch_size: (可选) int. 如果输入是 2D，这可能是 `from_tensor` 和 `to_tensor` 的 3D 版本的批大小
    from_seq_length: (可选) 如果输入是 2D，这可能是 `from_tensor` 的 3D 版本的序列长度
    to_seq_length: (可选) 如果输入是 2D，这可能是 `to_tensor` 的 3D 版本的序列长度

  返回：
    形状为 [batch_size, from_seq_length, num_attention_heads * size_per_head] 的 float 张量
    （如果 `do_return_2d_tensor` 为 true，则形状为 [batch_size * from_seq_length, num_attention_heads * size_per_head]）

  异常：
    ValueError: 任何参数或张量形状无效
  """


  def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                           seq_length, width):
    output_tensor = tf.reshape(
        input_tensor, [batch_size, seq_length, num_attention_heads, width])

    output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
    return output_tensor

  from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
  to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

  if len(from_shape) != len(to_shape):
    raise ValueError(
        "The rank of `from_tensor` must match the rank of `to_tensor`.")

  if len(from_shape) == 3:
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]
    to_seq_length = to_shape[1]
  elif len(from_shape) == 2:
    if (batch_size is None or from_seq_length is None or to_seq_length is None):
      raise ValueError(
          "When passing in rank 2 tensors to attention_layer, the values "
          "for `batch_size`, `from_seq_length`, and `to_seq_length` "
          "must all be specified.")

  # Scalar dimensions referenced here:
  #   B = batch size (number of sequences)
  #   F = `from_tensor` sequence length
  #   T = `to_tensor` sequence length
  #   N = `num_attention_heads`
  #   H = `size_per_head`

  from_tensor_2d = reshape_to_matrix(from_tensor)
  to_tensor_2d = reshape_to_matrix(to_tensor)

  # `query_layer` = [B*F, N*H]
  query_layer = tf.layers.dense(
      from_tensor_2d,
      num_attention_heads * size_per_head,
      activation=query_act,
      name="query",
      kernel_initializer=create_initializer(initializer_range))

  # `key_layer` = [B*T, N*H]
  key_layer = tf.layers.dense(
      to_tensor_2d,
      num_attention_heads * size_per_head,
      activation=key_act,
      name="key",
      kernel_initializer=create_initializer(initializer_range))

  # `value_layer` = [B*T, N*H]
  value_layer = tf.layers.dense(
      to_tensor_2d,
      num_attention_heads * size_per_head,
      activation=value_act,
      name="value",
      kernel_initializer=create_initializer(initializer_range))

  # `query_layer` = [B, N, F, H]
  query_layer = transpose_for_scores(query_layer, batch_size,
                                     num_attention_heads, from_seq_length,
                                     size_per_head)

  # `key_layer` = [B, N, T, H]
  key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                   to_seq_length, size_per_head)

  # Take the dot product between "query" and "key" to get the raw
  # attention scores.
  # `attention_scores` = [B, N, F, T]
  attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
  attention_scores = tf.multiply(attention_scores,
                                 1.0 / math.sqrt(float(size_per_head)))

  if attention_mask is not None:
    # `attention_mask` = [B, 1, F, T]
    attention_mask = tf.expand_dims(attention_mask, axis=[1])

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    attention_scores += adder

  # Normalize the attention scores to probabilities.
  # `attention_probs` = [B, N, F, T]
  attention_probs = tf.nn.softmax(attention_scores)

  # This is actually dropping out entire tokens to attend to, which might
  # seem a bit unusual, but is taken from the original Transformer paper.
  attention_probs = dropout(attention_probs, attention_probs_dropout_prob)

  # `value_layer` = [B, T, N, H]
  value_layer = tf.reshape(
      value_layer,
      [batch_size, to_seq_length, num_attention_heads, size_per_head])

  # `value_layer` = [B, N, T, H]
  value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

  # `context_layer` = [B, N, F, H]
  context_layer = tf.matmul(attention_probs, value_layer)

  # `context_layer` = [B, F, N, H]
  context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

  if do_return_2d_tensor:
    # `context_layer` = [B*F, N*H]
    context_layer = tf.reshape(
        context_layer,
        [batch_size * from_seq_length, num_attention_heads * size_per_head])
  else:
    # `context_layer` = [B, F, N*H]
    context_layer = tf.reshape(
        context_layer,
        [batch_size, from_seq_length, num_attention_heads * size_per_head])

  return context_layer


def transformer_model(input_tensor,
                      attention_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      intermediate_act_fn=gelu,
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False):
  """多头、多层 Transformer 编码器，基于 "Attention is All You Need" 论文

  这几乎是原始 Transformer 编码器的精确实现。

  参考原始论文：
  https://arxiv.org/abs/1706.03762

  也参考：
  https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py

  参数：
    input_tensor: 形状为 [batch_size, seq_length, hidden_size] 的 float 张量
    attention_mask: (可选) 形状为 [batch_size, seq_length, seq_length] 的 int32 张量，
      可被关注的位置为 1，不应被关注的位置为 0
    hidden_size: int. Transformer 的隐藏层大小
    num_hidden_layers: int. Transformer 中的层数（块数）
    num_attention_heads: int. Transformer 中的注意力头数
    intermediate_size: int. "中间"（即前馈）层的大小
    intermediate_act_fn: function. 应用于中间/前馈层输出的非线性激活函数
    hidden_dropout_prob: float. 隐藏层的 dropout 概率
    attention_probs_dropout_prob: float. 注意力概率的 dropout 概率
    initializer_range: float. 初始化器的范围（截断正态分布的标准差）
    do_return_all_layers: 是否同时返回所有层还是仅返回最终层

  返回：
    形状为 [batch_size, seq_length, hidden_size] 的 float 张量，Transformer 的最终隐藏层

  异常：
    ValueError: 张量形状或参数无效
  """

  if hidden_size % num_attention_heads != 0:
    raise ValueError(
        "The hidden size (%d) is not a multiple of the number of attention "
        "heads (%d)" % (hidden_size, num_attention_heads))

  attention_head_size = int(hidden_size / num_attention_heads)
  input_shape = get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  input_width = input_shape[2]

  # The Transformer performs sum residuals on all layers so the input needs
  # to be the same as the hidden size.
  if input_width != hidden_size:
    raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                     (input_width, hidden_size))

  # We keep the representation as a 2D tensor to avoid re-shaping it back and
  # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
  # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
  # help the optimizer.
  prev_output = reshape_to_matrix(input_tensor)

  all_layer_outputs = []
  for layer_idx in range(num_hidden_layers):
    with tf.variable_scope("layer_%d" % layer_idx):
      layer_input = prev_output

      with tf.variable_scope("attention"):
        attention_heads = []
        with tf.variable_scope("self"):
          attention_head = attention_layer(
              from_tensor=layer_input,
              to_tensor=layer_input,
              attention_mask=attention_mask,
              num_attention_heads=num_attention_heads,
              size_per_head=attention_head_size,
              attention_probs_dropout_prob=attention_probs_dropout_prob,
              initializer_range=initializer_range,
              do_return_2d_tensor=True,
              batch_size=batch_size,
              from_seq_length=seq_length,
              to_seq_length=seq_length)
          attention_heads.append(attention_head)

        attention_output = None
        if len(attention_heads) == 1:
          attention_output = attention_heads[0]
        else:
          # In the case where we have other sequences, we just concatenate
          # them to the self-attention head before the projection.
          attention_output = tf.concat(attention_heads, axis=-1)

        # Run a linear projection of `hidden_size` then add a residual
        # with `layer_input`.
        with tf.variable_scope("output"):
          attention_output = tf.layers.dense(
              attention_output,
              hidden_size,
              kernel_initializer=create_initializer(initializer_range))
          attention_output = dropout(attention_output, hidden_dropout_prob)
          attention_output = layer_norm(attention_output + layer_input)

      # The activation is only applied to the "intermediate" hidden layer.
      with tf.variable_scope("intermediate"):
        intermediate_output = tf.layers.dense(
            attention_output,
            intermediate_size,
            activation=intermediate_act_fn,
            kernel_initializer=create_initializer(initializer_range))

      # Down-project back to `hidden_size` then add the residual.
      with tf.variable_scope("output"):
        layer_output = tf.layers.dense(
            intermediate_output,
            hidden_size,
            kernel_initializer=create_initializer(initializer_range))
        layer_output = dropout(layer_output, hidden_dropout_prob)
        layer_output = layer_norm(layer_output + attention_output)
        prev_output = layer_output
        all_layer_outputs.append(layer_output)

  if do_return_all_layers:
    final_outputs = []
    for layer_output in all_layer_outputs:
      final_output = reshape_from_matrix(layer_output, input_shape)
      final_outputs.append(final_output)
    return final_outputs
  else:
    final_output = reshape_from_matrix(prev_output, input_shape)
    return final_output


def get_shape_list(tensor, expected_rank=None, name=None):
  """返回张量形状的列表，优先使用静态维度

  参数：
    tensor: 要查找形状的 tf.Tensor 对象
    expected_rank: (可选) int. `tensor` 的预期秩。如果指定了此参数且 `tensor` 的秩不同，将抛出异常
    name: 错误消息中张量的可选名称

  返回：
    张量形状的维度列表。所有静态维度将作为 python 整数返回，动态维度将作为 tf.Tensor 标量返回
  """

  if name is None:
    name = tensor.name

  if expected_rank is not None:
    assert_rank(tensor, expected_rank, name)

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape


def reshape_to_matrix(input_tensor):
  """将 >= 2 秩的张量重塑为 2 秩张量（即矩阵）

  参数：
    input_tensor: 输入张量，秩 >= 2

  返回：
    重塑后的 2 秩张量
  """

  ndims = input_tensor.shape.ndims
  if ndims < 2:
    raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                     (input_tensor.shape))
  if ndims == 2:
    return input_tensor

  width = input_tensor.shape[-1]
  output_tensor = tf.reshape(input_tensor, [-1, width])
  return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
  """将 2 秩张量重塑回其原始秩 >= 2 的张量

  参数：
    output_tensor: 2 秩张量
    orig_shape_list: 原始形状列表

  返回：
    重塑后的张量，秩与 orig_shape_list 相同
  """

  if len(orig_shape_list) == 2:
    return output_tensor

  output_shape = get_shape_list(output_tensor)

  orig_dims = orig_shape_list[0:-1]
  width = output_shape[-1]

  return tf.reshape(output_tensor, orig_dims + [width])


def assert_rank(tensor, expected_rank, name=None):
  """如果张量秩与预期秩不匹配，则引发异常

  参数：
    tensor: 要检查秩的 tf.Tensor
    expected_rank: Python 整数或整数列表，预期秩
    name: 错误消息中张量的可选名称

  异常：
    ValueError: 如果预期形状与实际形状不匹配
  """

  if name is None:
    name = tensor.name

  expected_rank_dict = {}
  if isinstance(expected_rank, six.integer_types):
    expected_rank_dict[expected_rank] = True
  else:
    for x in expected_rank:
      expected_rank_dict[x] = True

  actual_rank = tensor.shape.ndims
  if actual_rank not in expected_rank_dict:
    scope_name = tf.get_variable_scope().name
    raise ValueError(
        "For the tensor `%s` in scope `%s`, the actual rank "
        "`%d` (shape = %s) is not equal to the expected rank `%s`" %
        (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))
