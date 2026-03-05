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
"""标记化相关类和函数。

该模块实现了 BERT 模型使用的标记化功能，包括：
1. 基本标记化（BasicTokenizer）：处理标点符号分割、大小写转换等
2. WordPiece 标记化（WordpieceTokenizer）：将单词拆分为子词单元
3. 全标记化流程（FullTokenizer）：结合基本标记化和 WordPiece 标记化
4. 各种辅助函数：词汇表加载、ID 转换、字符串处理等

主要功能：
- 将原始文本转换为标记序列
- 处理多语言文本，包括中文
- 支持大小写转换
- 处理特殊字符和控制字符
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re
import unicodedata
import six
import tensorflow as tf


def validate_case_matches_checkpoint(do_lower_case, init_checkpoint):
  """检查大小写配置是否与检查点名称一致。

  验证用户指定的 do_lower_case 标志是否与预训练模型的大小写设置匹配。
  通过检查模型名称来推断其大小写设置。

  参数：
    do_lower_case: 是否使用小写
    init_checkpoint: 初始检查点路径

  异常：
    ValueError: 如果大小写配置与检查点不一致
  """


  # The casing has to be passed in by the user and there is no explicit check
  # as to whether it matches the checkpoint. The casing information probably
  # should have been stored in the bert_config.json file, but it's not, so
  # we have to heuristically detect it to validate.

  if not init_checkpoint:
    return

  m = re.match("^.*?([A-Za-z0-9_-]+)/bert_model.ckpt", init_checkpoint)
  if m is None:
    return

  model_name = m.group(1)

  lower_models = [
      "uncased_L-24_H-1024_A-16", "uncased_L-12_H-768_A-12",
      "multilingual_L-12_H-768_A-12", "chinese_L-12_H-768_A-12"
  ]

  cased_models = [
      "cased_L-12_H-768_A-12", "cased_L-24_H-1024_A-16",
      "multi_cased_L-12_H-768_A-12"
  ]

  is_bad_config = False
  if model_name in lower_models and not do_lower_case:
    is_bad_config = True
    actual_flag = "False"
    case_name = "lowercased"
    opposite_flag = "True"

  if model_name in cased_models and do_lower_case:
    is_bad_config = True
    actual_flag = "True"
    case_name = "cased"
    opposite_flag = "False"

  if is_bad_config:
    raise ValueError(
        "You passed in `--do_lower_case=%s` with `--init_checkpoint=%s`. "
        "However, `%s` seems to be a %s model, so you "
        "should pass in `--do_lower_case=%s` so that the fine-tuning matches "
        "how the model was pre-training. If this error is wrong, please "
        "just comment out this check." % (actual_flag, init_checkpoint,
                                          model_name, case_name, opposite_flag))


def convert_to_unicode(text):
  """将文本转换为 Unicode 格式（如果尚未是 Unicode）。

  假设输入文本是 UTF-8 编码，将其转换为 Unicode 字符串。
  支持 Python 2 和 Python 3。

  参数：
    text: 输入文本，可以是 str 或 bytes 类型

  返回值：
    Unicode 格式的文本

  异常：
    ValueError: 如果输入类型不受支持
  """

  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text.decode("utf-8", "ignore")
    elif isinstance(text, unicode):
      return text
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")


def printable_text(text):
  """返回适合打印或 `tf.logging` 的文本编码。

  将文本转换为适合打印或日志记录的格式，支持 Python 2 和 Python 3。

  参数：
    text: 输入文本，可以是 str 或 bytes 类型

  返回值：
    适合打印的文本

  异常：
    ValueError: 如果输入类型不受支持
  """


  # These functions want `str` for both Python2 and Python3, but in one case
  # it's a Unicode string and in the other it's a byte string.
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text
    elif isinstance(text, unicode):
      return text.encode("utf-8")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file):
  """加载词汇表文件到字典中。

  从指定的词汇表文件中加载词汇及其对应的索引，返回一个有序字典。

  参数：
    vocab_file: 词汇表文件路径

  返回值：
    有序字典，键为词汇，值为索引
  """

  vocab = collections.OrderedDict()
  index = 0
  with tf.gfile.GFile(vocab_file, "r") as reader:
    while True:
      token = convert_to_unicode(reader.readline())
      if not token:
        break
      token = token.strip()
      vocab[token] = index
      index += 1
  return vocab


def convert_by_vocab(vocab, items):
  """使用词汇表转换标记序列或 ID 序列。

  根据提供的词汇表，将标记序列转换为 ID 序列，或反之。

  参数：
    vocab: 词汇表字典
    items: 要转换的标记或 ID 序列

  返回值：
    转换后的序列
  """

  output = []
  for item in items:
    output.append(vocab[item])
  return output


def convert_tokens_to_ids(vocab, tokens):
  return convert_by_vocab(vocab, tokens)


def convert_ids_to_tokens(inv_vocab, ids):
  return convert_by_vocab(inv_vocab, ids)


def whitespace_tokenize(text):
  """对文本进行基本的空白字符清理和分割。

  移除文本两端的空白字符，然后按空白字符分割文本。

  参数：
    text: 输入文本

  返回值：
    分割后的标记列表
  """

  text = text.strip()
  if not text:
    return []
  tokens = text.split()
  return tokens


class FullTokenizer(object):
  """运行端到端的标记化流程。

  结合 BasicTokenizer 和 WordpieceTokenizer，执行完整的标记化过程。

  属性：
    vocab: 词汇表字典
    inv_vocab: 反向词汇表字典（ID 到标记的映射）
    basic_tokenizer: BasicTokenizer 实例
    wordpiece_tokenizer: WordpieceTokenizer 实例

  方法：
    tokenize: 将文本转换为标记序列
    convert_tokens_to_ids: 将标记序列转换为 ID 序列
    convert_ids_to_tokens: 将 ID 序列转换为标记序列
  """


  def __init__(self, vocab_file, do_lower_case=True):
    self.vocab = load_vocab(vocab_file)
    self.inv_vocab = {v: k for k, v in self.vocab.items()}
    self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
    self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

  def tokenize(self, text):
    split_tokens = []
    for token in self.basic_tokenizer.tokenize(text):
      for sub_token in self.wordpiece_tokenizer.tokenize(token):
        split_tokens.append(sub_token)

    return split_tokens

  def convert_tokens_to_ids(self, tokens):
    return convert_by_vocab(self.vocab, tokens)

  def convert_ids_to_tokens(self, ids):
    return convert_by_vocab(self.inv_vocab, ids)


class BasicTokenizer(object):
  """运行基本标记化（标点符号分割、大小写转换等）。

  执行基本的文本处理，包括清理文本、处理中文字符、分割标点符号、大小写转换等。

  属性：
    do_lower_case: 是否将文本转换为小写

  方法：
    tokenize: 将文本转换为标记序列
    _run_strip_accents: 移除文本中的重音符号
    _run_split_on_punc: 按标点符号分割文本
    _tokenize_chinese_chars: 处理中文字符
    _is_chinese_char: 检查字符是否为中文字符
    _clean_text: 清理文本中的无效字符和空白字符
  """


  def __init__(self, do_lower_case=True):
    """构造 BasicTokenizer 实例。

    参数：
      do_lower_case: 是否将输入文本转换为小写
    """

    self.do_lower_case = do_lower_case

  def tokenize(self, text):
    """对文本进行标记化处理。

    执行以下步骤：
    1. 将文本转换为 Unicode 格式
    2. 清理文本中的无效字符和空白字符
    3. 处理中文字符（在中文字符周围添加空白）
    4. 按空白字符分割文本
    5. 如果需要，将文本转换为小写并移除重音符号
    6. 按标点符号分割文本
    7. 再次按空白字符分割文本

    参数：
      text: 输入文本

    返回值：
      标记化后的标记列表
    """

    text = convert_to_unicode(text)
    text = self._clean_text(text)

    # This was added on November 1st, 2018 for the multilingual and Chinese
    # models. This is also applied to the English models now, but it doesn't
    # matter since the English models were not trained on any Chinese data
    # and generally don't have any Chinese data in them (there are Chinese
    # characters in the vocabulary because Wikipedia does have some Chinese
    # words in the English Wikipedia.).
    text = self._tokenize_chinese_chars(text)

    orig_tokens = whitespace_tokenize(text)
    split_tokens = []
    for token in orig_tokens:
      if self.do_lower_case:
        token = token.lower()
        token = self._run_strip_accents(token)
      split_tokens.extend(self._run_split_on_punc(token))

    output_tokens = whitespace_tokenize(" ".join(split_tokens))
    return output_tokens

  def _run_strip_accents(self, text):
    """移除文本中的重音符号。

    使用 Unicode 规范化将重音符号从文本中移除。

    参数：
      text: 输入文本

    返回值：
      移除重音符号后的文本
    """

    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
      cat = unicodedata.category(char)
      if cat == "Mn":
        continue
      output.append(char)
    return "".join(output)

  def _run_split_on_punc(self, text):
    """按标点符号分割文本。

    将文本中的标点符号分割成单独的标记。

    参数：
      text: 输入文本

    返回值：
      分割后的标记列表
    """

    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
      char = chars[i]
      if _is_punctuation(char):
        output.append([char])
        start_new_word = True
      else:
        if start_new_word:
          output.append([])
        start_new_word = False
        output[-1].append(char)
      i += 1

    return ["".join(x) for x in output]

  def _tokenize_chinese_chars(self, text):
    """在中文字符周围添加空白字符。

    为了正确处理中文文本，在每个中文字符周围添加空白字符，
    这样后续的标记化过程可以将每个中文字符视为一个单独的标记。

    参数：
      text: 输入文本

    返回值：
      处理后的文本
    """

    output = []
    for char in text:
      cp = ord(char)
      if self._is_chinese_char(cp):
        output.append(" ")
        output.append(char)
        output.append(" ")
      else:
        output.append(char)
    return "".join(output)

  def _is_chinese_char(self, cp):
    """检查字符是否为中文字符。

    根据 Unicode 编码范围判断字符是否为中文字符。
    包括 CJK 统一表意文字、扩展 A 到 F 等范围。

    参数：
      cp: 字符的 Unicode 编码

    返回值：
      如果是中文字符，返回 True，否则返回 False
    """

    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
        (cp >= 0x3400 and cp <= 0x4DBF) or  #
        (cp >= 0x20000 and cp <= 0x2A6DF) or  #
        (cp >= 0x2A700 and cp <= 0x2B73F) or  #
        (cp >= 0x2B740 and cp <= 0x2B81F) or  #
        (cp >= 0x2B820 and cp <= 0x2CEAF) or
        (cp >= 0xF900 and cp <= 0xFAFF) or  #
        (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
      return True

    return False

  def _clean_text(self, text):
    """清理文本中的无效字符和空白字符。

    移除文本中的无效字符（如控制字符），并将各种空白字符统一为空格。

    参数：
      text: 输入文本

    返回值：
      清理后的文本
    """

    output = []
    for char in text:
      cp = ord(char)
      if cp == 0 or cp == 0xfffd or _is_control(char):
        continue
      if _is_whitespace(char):
        output.append(" ")
      else:
        output.append(char)
    return "".join(output)


class WordpieceTokenizer(object):
  """运行 WordPiece 标记化。

  使用贪心最长匹配算法将单词拆分为子词单元。

  属性：
    vocab: 词汇表字典
    unk_token: 未知标记
    max_input_chars_per_word: 每个单词的最大字符数

  方法：
    tokenize: 将文本转换为 WordPiece 标记序列
  """

  def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
    """构造 WordpieceTokenizer 实例。

    参数：
      vocab: 词汇表字典
      unk_token: 未知标记
      max_input_chars_per_word: 每个单词的最大字符数
    """
    self.vocab = vocab
    self.unk_token = unk_token
    self.max_input_chars_per_word = max_input_chars_per_word

  def tokenize(self, text):
    """将文本转换为 WordPiece 标记序列。

    使用贪心最长匹配算法，根据给定的词汇表进行标记化。

    例如：
      输入 = "unaffable"
      输出 = ["un", "##aff", "##able"]

    参数：
      text: 单个标记或空白分隔的标记。应该已经通过 `BasicTokenizer` 处理。

    返回值：
      WordPiece 标记列表。
    """


    text = convert_to_unicode(text)

    output_tokens = []
    for token in whitespace_tokenize(text):
      chars = list(token)
      if len(chars) > self.max_input_chars_per_word:
        output_tokens.append(self.unk_token)
        continue

      is_bad = False
      start = 0
      sub_tokens = []
      while start < len(chars):
        end = len(chars)
        cur_substr = None
        while start < end:
          substr = "".join(chars[start:end])
          if start > 0:
            substr = "##" + substr
          if substr in self.vocab:
            cur_substr = substr
            break
          end -= 1
        if cur_substr is None:
          is_bad = True
          break
        sub_tokens.append(cur_substr)
        start = end

      if is_bad:
        output_tokens.append(self.unk_token)
      else:
        output_tokens.extend(sub_tokens)
    return output_tokens


def _is_whitespace(char):
  """检查字符是否为空白字符。

  判断字符是否为空白字符，包括空格、制表符、换行符、回车符以及 Unicode 空格类别。

  参数：
    char: 输入字符

  返回值：
    如果是空白字符，返回 True，否则返回 False
  """

  # \t, \n, and \r are technically contorl characters but we treat them
  # as whitespace since they are generally considered as such.
  if char == " " or char == "\t" or char == "\n" or char == "\r":
    return True
  cat = unicodedata.category(char)
  if cat == "Zs":
    return True
  return False


def _is_control(char):
  """检查字符是否为控制字符。

  判断字符是否为控制字符，但将制表符、换行符和回车符视为空白字符而非控制字符。

  参数：
    char: 输入字符

  返回值：
    如果是控制字符，返回 True，否则返回 False
  """

  # These are technically control characters but we count them as whitespace
  # characters.
  if char == "\t" or char == "\n" or char == "\r":
    return False
  cat = unicodedata.category(char)
  if cat in ("Cc", "Cf"):
    return True
  return False


def _is_punctuation(char):
  """检查字符是否为标点符号。

  判断字符是否为标点符号，包括 ASCII 标点符号和 Unicode 标点符号类别。

  参数：
    char: 输入字符

  返回值：
    如果是标点符号，返回 True，否则返回 False
  """

  cp = ord(char)
  # We treat all non-letter/number ASCII as punctuation.
  # Characters such as "^", "$", and "`" are not in the Unicode
  # Punctuation class but we treat them as punctuation anyways, for
  # consistency.
  if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
      (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
    return True
  cat = unicodedata.category(char)
  if cat.startswith("P"):
    return True
  return False
