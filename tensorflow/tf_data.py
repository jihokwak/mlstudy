import os
import tensorflow as tf
import numpy as np
from tensorflow.python.keras import preprocessing

#전처리
samples = ['너 오늘 이뻐 보인다',
           '나는 오늘 기분이 더러워',
           '끝내주는데, 좋은 일이 있나봐',
           '나 좋은 일이 생겼어',
           '아 오늘 진짜 짜증나',
           '환상적인데, 정말 좋은거 같아']
label = [[1],[0],[1],[1],[0],[1]]


tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(samples) # 토크나이저 피팅
sequences = tokenizer.texts_to_sequences(samples) #텍스트 수치화
word_index = tokenizer.word_index # 단어 인덱스


#기본 출력
dataset = tf.data.Dataset.from_tensor_slices((sequences, label)) #시퀀스데이터와 레이블 묶어 조각만들기
iterator = dataset.make_one_shot_iterator() #이터레이터로 변환
next_data = iterator.get_next() #하나씩 데이터 추출

with tf.Session() as sess :
    while True :
        try :
            print(sess.run(next_data))
        except tf.errors.OutOfRangeError:
            break

#배치단위 출력
BATCH_SIZE = 2

dataset = tf.data.Dataset.from_tensor_slices((sequences, label))
dataset = dataset.batch(BATCH_SIZE)
iterator = dataset.make_one_shot_iterator()
next_data = iterator.get_next()

with tf.Session() as sess:
    while True :
        try :
            print(sess.run(next_data))
        except tf.errors.OutOfRangeError:
            break

#셔플
dataset = tf.data.Dataset.from_tensor_slices((sequences, label))
dataset = dataset.shuffle(len(sequences))
iterator = dataset.make_one_shot_iterator()
next_data = iterator.get_next()

with tf.Session() as sess :

    while True :
        try :
            print(sess.run(next_data))

        except tf.errors.OutOfRangeError :
            break

#에포크 설정으로 데이터 반복 호출
EPOCHS = 2

dataset = tf.data.Dataset.from_tensor_slices((sequences, label))
dataset = dataset.repeat(EPOCHS)
iterator = dataset.make_one_shot_iterator()
next_data = iterator.get_next()

with tf.Session() as sess :
    while True :
        try :
            print(sess.run(next_data))
        except tf.errors.OutOfRangeError :
            break

#매핑
def mapping_fn(X, Y=None) :
    input = {'x' : X}
    label = Y
    return input, label

dataset = tf.data.Dataset.from_tensor_slices((sequences, label))
dataset = dataset.map(mapping_fn)
iterator = dataset.make_one_shot_iterator()
next_data = iterator.get_next()

with tf.Session() as sess :
    while True :
        try :
            print(sess.run(next_data))
        except tf.errors.OutOfRangeError :
            break


#배치, 셔플. 반복, 매핑 종합
BATCH_SIZE = 2
EPOCHS = 2

def mapping_fn(X, Y=None) :
    input = {'x':X}
    label = Y
    return input, label

next_data = tf.data.Dataset.from_tensor_slices((sequences, label)) \
            .map(mapping_fn) \
            .shuffle(len(sequences)) \
            .batch(BATCH_SIZE) \
            .repeat(EPOCHS) \
            .make_one_shot_iterator() \
            .get_next()

with tf.Session() as sess :
    while True :
        try :
            print(sess.run(next_data))
        except tf.errors.OutOfRangeError :
            break

            