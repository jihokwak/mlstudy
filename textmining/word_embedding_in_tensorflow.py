import numpy as np

dataset = ['I like going to the cinema.',
           'Mum is going to cook spinach tonight.',
           'Do you think they are going to win the match ?',
           'Are you going to the United States ?',
           "He doesn't like playing video games."
          ]


vocab = []
symbols = {0:'PAD', 1:'UNK'}

for sentence in dataset:
    for word in sentence.split():
        vocab.append(word.lower())

vocab = list(set(vocab))

#Loag glove or any word_embedding
def word_embedding_matrix(embedding_path, vocab, dim):
    with open(embedding_path, "r", encoding='utf8') as f:
        word_vocab = []
        embedding_matrix = []
        word_vocab.extend(['PAD','UNK'])
        embedding_matrix.append(np.random.uniform(-1.0, 1.0, (1,100))[0])
        embedding_matrix.append(np.random.uniform(-1.0, 1.0, (1,100))[0])

        for line in f :
            if line.split()[0] in vocab:
                word_vocab.append(line.split()[0])
                embedding_matrix.append([float(i) for i in line.split()[1:]])
    return {'word_vocab' : word_vocab, 'Embedding_matrix':np.reshape(embedding_matrix, [-1,dim]).astype(np.float32)}

filename = "D:/Kwak/Doc/git/kmu2/data/glove/glove.6B.100d.txt"
print(word_embedding_matrix(filename, vocab, 100)['Embedding_matrix'][2])

#build int_to_vocab and vocab_to_int
matrix = word_embedding_matrix(filename, vocab, 100)
load_embedding_matrix = matrix['Embedding_matrix']
shape_word_vocab = matrix['word_vocab']

int_to_vocab = {}

for index_no, word in enumerate(shape_word_vocab):
    int_to_vocab[index_no] = word

int_to_vocab.update(symbols)

vocab_to_int = {val:key for key, val in int_to_vocab.items()}

#Encode the sentences using vocab
encoded_data = []

for sentence in dataset:
    sentence_ = []
    for word in sentence.split():
        if word.lower() in vocab_to_int:
            sentence_.append(vocab_to_int[word.lower()])
    encoded_data.append(sentence_)

print(encoded_data)

#check
for i in encoded_data:
    print([int_to_vocab[j] for j in i])

#Using Embedding matrix in Tensorflow
import tensorflow as tf
tf.reset_default_graph()
sentences = tf.placeholder(tf.int32, shape=[None, None])
Word_embedding = tf.get_variable(name='Word_embedding',
                                 shape=[24, 100],
                                 initializer=tf.constant_initializer(np.array(load_embedding_matrix)),
                                 trainable=False)

embedding_lookup = tf.nn.embedding_lookup(Word_embedding, sentences)

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
    for embedding_vactor in encoded_data:
        print(sess.run(embedding_lookup, feed_dict={sentences:[embedding_vactor]}))

def word_embedding_matrix(path, dim) :
    word_id = {}
    with open(path, 'r', encoding='utf-8') as fp:
        vocabulary = []
        embedding_matrix = [np.random.uniform(-1.0, 1.0, (dim, 1))]
        for line in fp.readlines() :
            if line.split()[0] in vocab:
                vocabulary.append(line.split()[0])
                embedding_matrix.append(np.reshape(np.array(line.split()[1:]), (dim, 1)))
    word_id['PAD'] = 0
    word_id['UNK'] = len(vocabulary) + 1
    embedding_matrix.append(np.random.uniform(-1.0, 1.0, (dim, 1)))
    embedding_matrix = np.reshape(embedding_matrix, [-1, dim])
    embedding_matrix = embedding_matrix.astype(np.float32)

    return embedding_matrix, vocabulary

embedding_matrix, words2ids = word_embedding_matrix(filename,100)
