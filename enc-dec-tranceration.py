#!/usr/bin/python
# coding: UTF-8
USE_GPU = True
DEBUG = True

import datetime
from argparse import ArgumentParser

from Vocabulary import Vocabulary
import random
from chainer import Chain, cuda, Variable, links, optimizers, serializers
from chainer import functions as F

if USE_GPU:
  import cupy as xp
else:
  import numpy as xp

if not DEBUG:
  import os
  os.environ["CHAINER_TYPE_CHECK"] = "0" #type_checkをしない

def parse_args():
  p = ArgumentParser(description='Encoder-decoder neural machine translation')
  p.add_argument("model", type=str)
  p.add_argument("mode", type=str, help="train mode or test mode")
  p.add_argument("source", type=str)
  p.add_argument("target", type=str)
  p.add_argument("s_test", type=str)
  p.add_argument("t_test", type=str)
  p.add_argument("s_vocab", type=str)
  p.add_argument("t_vocab", type=str)
  p.add_argument("--embed_size", dest="embed_size", type=int, default=100)
  p.add_argument("--hidden_size", dest="hidden_size", type=int, default=200,
                        help="the number of cells at hidden layer")
  p.add_argument("--epochs", dest="epochs", type=int, default=1)
  p.add_argument("--batch_size", dest="batch_size", type=int, default=100)
  return p.parse_args()  

## todo:ファイル名がハードコーティングなのを直す
def trace(*args):
  output_file = 'output_1.txt'
  with open(output_file, 'a') as fp:
    print('[', datetime.datetime.now(), ']', *args, file=fp)

def my_zeros(shape, dtype):
  return Variable(xp.zeros(shape, dtype=dtype))

def my_array(array, dtype):
  return Variable(xp.array(array, dtype=dtype))



def load_vocab(file):
  v = Vocabulary()
  v.load(file)
  return v

def load_input(file, vocab):
  with open(file, "r") as f:
    for line in f:
      line = line.strip().split()
      yield [vocab.w2id(e) for e in line]

## todo: test時（学習中のvalidation時ではなく）のときにすべて出力するようにする
def show_outputs(pre_trained, trained, sources, targets, outputs, s_vocab, t_vocab, train):
  if int(pre_trained/20000) != int(trained/20000):
    trace('------- trained:', trained, ' --------')
    for s, t, o in zip(sources, targets, outputs):
      trace('source:', " ".join([s_vocab.id2w(e) for e in s if e != -1]))
      trace('target:', " ".join([t_vocab.id2w(e) for e in t if e != -1]))
      trace('outputs:', " ".join([t_vocab.id2w(e) for e in o if e != -1]))
      trace('----')   
  elif train == False and int(pre_trained/200) != int(trained/200):
    trace('------- tested:', trained, ' --------')
    for s, t, o in zip(sources, targets, outputs):
      trace('t_source:', " ".join([s_vocab.id2w(e) for e in s if e != -1]))
      trace('t_target:', " ".join([t_vocab.id2w(e) for e in t if e != -1]))
      trace('t_outputs:', " ".join([t_vocab.id2w(e) for e in o if e != -1]))
      trace('----')
  return



def batch(gen, batch_size):
  batch = []
  for e in gen:
    batch.append(e)
    if len(batch) == batch_size:
      yield batch
      batch = []
  if batch:
    yield batch

def parallel_batch(gen, batch_size):
  batch = [[],[]]
  for e in gen:
    batch[0].append(e[0])
    batch[1].append(e[1])
    if len(batch[0]) == batch_size:
      yield batch
      batch = [[],[]]
  if batch != [[],[]]:
    yield batch

def sort_gen(s_gen, t_gen, sort_size):
  gen1 = batch(s_gen, sort_size)
  gen2 = batch(t_gen, sort_size)
  for e1, e2 in zip(gen1, gen2):
    for x in sorted(zip(e1, e2), key=lambda x: (len(x[1]), len(x[0]))):
      yield list(x)

def suffle_gen(gen, suffle_size):
  gen = batch(gen, suffle_size)
  for e in gen:
    for i in sorted(e, key=lambda i: random.random()):
      yield i

def generater(s_gen, t_gen, batch_size, sort_block=100):
  sorted_gen = sort_gen(s_gen, t_gen, batch_size*sort_block)
  batch_gen = parallel_batch(sorted_gen, batch_size)
  gen = suffle_gen(batch_gen, sort_block)
  return gen

def fill_batch(batch, token='</s>'):
    max_len = max(len(x) for x in batch)
    return [x + [token] * (max_len - len(x) + 1) for x in batch]




class Encoder(Chain):
  def __init__(self, embed_size, hidden_size, source_vocab):
    super(Encoder, self).__init__(
      word_id_2_embed=F.EmbedID(source_vocab, embed_size, ignore_label=-1),
      embed_2_lstm_input=F.Linear(embed_size, hidden_size*4),
      pre_hidden_2_lstm_input=F.Linear(hidden_size, hidden_size*4),
      )

  def __call__(self, x, c, p):
    word_embed = self.word_id_2_embed(x)
    lstm_input = self.embed_2_lstm_input(word_embed) + self.pre_hidden_2_lstm_input(p)
    c, p = F.lstm(c, lstm_input)
    return c, p

class Decoder(Chain):
  def __init__(self, embed_size, hidden_size, target_vocab):
    super(Decoder, self).__init__(
      word_id_2_embed=F.EmbedID(target_vocab, embed_size, ignore_label=-1),
      embed_2_lstm_input=F.Linear(embed_size, hidden_size*4),
      pre_hidden_2_lstm_input=F.Linear(hidden_size, hidden_size*4),
      hidden_2_word_id=F.Linear(hidden_size, target_vocab),
      )

  def __call__(self, x, c, q):
    word_embed = self.word_id_2_embed(x)
    lstm_input = self.embed_2_lstm_input(word_embed) + self.pre_hidden_2_lstm_input(q)
    c, q = F.lstm(c, lstm_input)
    y = self.hidden_2_word_id(q)
    return c, q, y

## todo: h, cを外部からいじれるようにする（今は1-best、beam-searchを使わないことが前提の書き方になっている）
class EncoderDecoder(Chain):
  def __init__(self, embed_size, hidden_size, s_vocab_size, t_vocab_size):
    super(EncoderDecoder, self).__init__(
      enc = Encoder(embed_size, hidden_size, s_vocab_size),
      dec = Decoder(embed_size, hidden_size, t_vocab_size),
    )
    self.embed_size = embed_size
    self.hidden_size = hidden_size

  def reset_state(self, batch_size):
    self.zerograds()
    self.c = my_zeros((batch_size, self.hidden_size), xp.float32)
    self.h = my_zeros((batch_size, self.hidden_size), xp.float32)

  def encode(self, x):
    self.c, self.h = self.enc(x, self.c, self.h)

  def decode(self, x):
    self.c, self.h, y = self.dec(x, self.c, self.h)
    return y

  def save_spec(self, filename):
    with open(filename, 'w') as fp:
      print(self.embed_size, file=fp)
      print(self.hidden_size, file=fp)

  @staticmethod
  def load_spec(filename, s_vocab_size, t_vocab_size):
    with open(filename) as fp:
      embed_size = int(next(fp))
      hidden_size = int(next(fp))
      return EncoderDecoder(embed_size, hidden_size, s_vocab_size, t_vocab_size)


def forward(model, source_batch, target_batch, batch_size, is_train=True, t_EOS_id=0):
  ## initalize
  source_len = len(source_batch[0])
  target_len = len(target_batch[0])
  output = [[] for _ in range(batch_size)]
  model.reset_state(batch_size)

  ## encode
  for index in reversed(range(source_len)):
    x = my_array([source_batch[i][index] for i in range(batch_size)], xp.int32)
    model.encode(x)

  ## decode
  if is_train:
    loss = my_zeros((), xp.float32)
    x = my_array([t_EOS_id for _ in range(batch_size)], xp.int32)
    for index in range(target_len):
      y = model.decode(x)
      t = my_array([target_batch[i][index] for i in range(batch_size)], xp.int32)
      loss += F.softmax_cross_entropy(y, t)
      predict_words = cuda.to_cpu(y.data.argmax(1))
      x = t # 正解を入力として使う

      for i in range(batch_size):
        output[i].append(predict_words[i])
    return loss, output

  else:
    loss = my_zeros((), xp.float32)
    x = my_array([t_EOS_id for _ in range(batch_size)], xp.int32)
    for index in range(target_len):
      y = model.decode(x)
      t = my_array([target_batch[i][index] for i in range(batch_size)], xp.int32)
      loss += F.softmax_cross_entropy(y, t)
      predict_words = cuda.to_cpu(y.data.argmax(1))
      x = my_array(predict_words, xp.int32) # 違いは、入力として予測値を使うか正解を使うか

      for i in range(batch_size):
        output[i].append(predict_words[i])
    return loss, output

  """ 本当のテスト時（正解データが無い時用）
  else: 
    x = my_array([t_EOS_id for _ in range(batch_size)], xp.int32)
    for index in range(target_len):
      y = model.decode(x)
      predict_words = cuda.to_cpu(y.data.argmax(1))
      x = my_array(predict_words, xp.int32)

      for i in range(batch_size):
        output[i].append(predict_words[i])
    return output
  """

def epoch_loop(args, encdec, epochs, s_file, t_file, s_vocab, t_vocab, train=True):
  """ 学習orテスト時のメイン動作部分 """
  # 初期化
  t_EOS_id = t_vocab.w2id('<EOS>')
  # train時のみoptimaizerの準備  
  if train:
    opt = optimizers.Adam()
    opt.setup(encdec)
  # GPU時の設定 (0番目のgpuを使用する、という意味)
  if USE_GPU: 
    encdec.to_gpu(0)

  for epoch in range(epochs):
    trace('epoch:', epoch)
    # 各epochでの初期化
    trained = 0
    sum_loss = 0

    # generaterの準備
    s_gen = load_input(s_file, s_vocab)
    t_gen = load_input(t_file, t_vocab)
    gen = generater(s_gen, t_gen, args.batch_size)

    for s, t in gen: # s, tはバッチサイズ文のデータ
      # EOSをtarget文の最後に追加
      for i in range(len(t)):
        t[i] = t[i] + [t_EOS_id]

      # 長さの違う部分を-1で埋める（-1で埋めると、word embedが全て0になる、かつlossが計算されない）
      s = fill_batch(s, token=-1)
      t = fill_batch(t, token=-1)

      cur_batch_size = len(s)
      loss, outputs = forward(encdec, s, t, cur_batch_size, train, t_EOS_id)
      sum_loss += loss.data*cur_batch_size
      # 重みの更新
      if train:
        loss.backward()
        opt.update()

      trained += cur_batch_size
      if args.mode == 'train':
        show_outputs(trained - cur_batch_size, trained, s[:5], t[:5], outputs[:5], s_vocab, t_vocab, train)
      else:
        show_outputs(trained - cur_batch_size, trained, s, t, outputs, s_vocab, t_vocab, train)        

    # 後処理
    if train:
      trace('*** end_epoch:', epoch, ' ***')
      trace('loss:', sum_loss)
      trace('******')
    else:
      trace('*** end_test:', epoch, ' ***')
      trace('test_loss:', sum_loss)
      trace('******')

    #if train:
    if train and (epoch+1)%3==0:
      trace('saving model ...')
      prefix = args.model + '.%03.d' % (epoch + 1)
      encdec.save_spec(prefix + '.spec')
      serializers.save_hdf5(prefix + '.weights', encdec)
      serializers.save_hdf5(prefix + '.opt', opt)

      # テストをして汎化性能をチェック
      epoch_loop(args, encdec, 1, args.s_test, args.t_test, s_vocab, t_vocab, train=False)

  return

def main():
  args = parse_args()

  trace('load vocabulary.....')
  source_vocab = load_vocab(args.s_vocab)
  target_vocab = load_vocab(args.t_vocab)

  if args.mode == 'train':
    trace('make model.....')
    encdec = EncoderDecoder(args.embed_size, args.hidden_size, source_vocab.size, target_vocab.size)
    epoch_loop(args, encdec, args.epochs, args.source, args.target, source_vocab, target_vocab, train=True)
  else:
    trace('load model.....')
    encdec = EncoderDecoder.load_spec(args.model + '.spec', source_vocab.size, target_vocab.size)
    serializers.load_hdf5(args.model + '.weights', encdec)
    epoch_loop(args, encdec, 1, args.s_test, args.t_test, source_vocab, target_vocab, train=False)

if __name__ == '__main__':
  main()







