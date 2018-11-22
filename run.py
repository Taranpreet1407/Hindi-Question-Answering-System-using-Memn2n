from __future__ import absolute_import
from __future__ import print_function
import os
import re

from sklearn.model_selection import train_test_split
from sklearn import metrics
from memn2n import MemN2N
from itertools import chain
from six.moves import range, reduce

import tensorflow as tf
import numpy as np
import sys

def get_labels(trainA, testA, valA):
    train_labels = np.argmax(trainA, axis=1)
    test_labels = np.argmax(testA, axis=1)
    val_labels = np.argmax(valA, axis=1)

    return train_labels, test_labels, val_labels


def get_batches(n_train, FLAGS):
    batch_size = FLAGS.batch_size
    batches = zip(range(0, n_train - batch_size, batch_size), range(batch_size, n_train, batch_size))
    batches = [(start, end) for start, end in batches]
    
    return batch_size, batches


def get_shapes(testS, trainS, valS):
    n_test = testS.shape[0]
    n_train = trainS.shape[0]
    n_val = valS.shape[0]

    return n_test, n_train, n_val


def load_task(data_dir, task_id, only_supporting=False):
    
    assert task_id > 0 and task_id < 21

    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    s = 'qa{}_'.format(task_id)
    train_file = [f for f in files if s in f and 'train' in f][0]
    test_file = [f for f in files if s in f and 'test' in f][0]
    train_data = get_stories(train_file, only_supporting)
    test_data = get_stories(test_file, only_supporting)
    return train_data, test_data

def get_total_cost(batches, model, trainS, trainQ, trainA, lr):
    total_cost = 0.0
    for start, end in batches:
        s = trainS[start:end]
        q = trainQ[start:end]
        a = trainA[start:end]
        cost_t = model.train(s, q, a, lr)
        total_cost += cost_t

    return total_cost

def tokenize(sent):
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def get_updated_learning_rate(t, FLAGS):
    # Stepped learning rate
    if t - 1 <= FLAGS.anneal_stop_epoch:
        anneal = 2.0 ** ((t - 1) // FLAGS.anneal_rate)
    else:
        anneal = 2.0 ** (FLAGS.anneal_stop_epoch // FLAGS.anneal_rate)
    return (FLAGS.learning_rate / anneal)


def parse_stories(lines, only_supporting=False):
    data = []
    story = []
    for line in lines:
        line = str.lower(line)
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line: # question
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            #a = tokenize(a)
            # answer is one vocab word even if it's actually multiple words
            a = [a]
            substory = None

            # remove question marks
            if q[-1] == "?":
                q = q[:-1]

            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]

            data.append((substory, q, a))
            story.append('')
        else: # regular sentence
            # remove periods
            sent = tokenize(line)
            if sent[-1] == ".":
                sent = sent[:-1]
            story.append(sent)
    return data


def get_stories(f, only_supporting=False):
    with open(f) as f:
        return parse_stories(f.readlines(), only_supporting=only_supporting)


def vectorize_data(data, word_idx, sentence_size, memory_size):
    S = []
    Q = []
    A = []
    for story, query, answer in data:
        ss = []
        for i, sentence in enumerate(story, 1):
            ls = max(0, sentence_size - len(sentence))
            ss.append([word_idx[w] for w in sentence] + [0] * ls)

        ss = ss[::-1][:memory_size][::-1]
        for i in range(len(ss)):
            ss[i][-1] = len(word_idx) - memory_size - i + len(ss)

        # pad to memory_size
        lm = max(0, memory_size - len(ss))
        for _ in range(lm):
            ss.append([0] * sentence_size)
        lq = max(0, sentence_size - len(query))
        q = [word_idx[w] for w in query] + [0] * lq
        y = np.zeros(len(word_idx) + 1) # 0 is reserved for nil word
        for a in answer:
            y[word_idx[a]] = 1

        S.append(ss)
        Q.append(q)
        A.append(y)

    return np.array(S), np.array(Q), np.array(A)


def get_train_predictions(FLAGS, trainS, trainQ, n_train, batch_size, model):
    train_predictions = []
    for start in range(0, n_train, batch_size):
        end = start + batch_size
        s = trainS[start:end]
        q = trainQ[start:end]
        pred = model.predict(s, q)
        train_predictions += list(pred)

    return train_predictions


def printTests(l,r, test, test_predictions, test_labels, inverted_word_idx):
    for i in range(l,r):
        story = test[i][0]
        query = test[i][1]
        ans = test[i][2][0]

        for sent in story:
            for word in sent:
                print(word, end=' ')
            print('')
        print('-->',end='')
        for word in query:
            print(word, end=' ')
        print('')
        print("Actual : "+ans)
        print("Prediction : "+inverted_word_idx[test_predictions[i]])
        print('')


def get_sizes(data):
    max_story_size = max(map(len, (s for s, _, _ in data)))
    mean_story_size = int(np.mean([ len(s) for s, _, _ in data ]))
    sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in data)))
    query_size = max(map(len, (q for _, q, _ in data)))

    return max_story_size, mean_story_size, sentence_size, query_size


def main(task_id):
    tf.flags.DEFINE_integer("hops", 3, "")
    tf.flags.DEFINE_integer("epochs", 100, "")
    tf.flags.DEFINE_integer("embedding_size", 100, "")
    tf.flags.DEFINE_float("learning_rate", 0.01, "")
    tf.flags.DEFINE_float("anneal_rate", 25, "")
    tf.flags.DEFINE_float("anneal_stop_epoch", 100, "")
    tf.flags.DEFINE_float("max_grad_norm", 40.0, "")
    tf.flags.DEFINE_integer("random_state", None, "")
    tf.flags.DEFINE_integer("evaluation_interval", 10, "")
    tf.flags.DEFINE_integer("batch_size", 32, "")
    tf.flags.DEFINE_integer("memory_size", 50, "")
    tf.flags.DEFINE_integer("task_id", task_id, "")
    tf.flags.DEFINE_string("data_dir", "data/tasks_1-20_v1-2/hn/", "")
    FLAGS = tf.flags.FLAGS

    print("Started Task:", FLAGS.task_id)

    train, test = load_task(FLAGS.data_dir, FLAGS.task_id)
    data = train + test
    vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data)))

    max_story_size, mean_story_size, sentence_size, query_size = get_sizes(data)
    memory_size = min(FLAGS.memory_size, max_story_size)
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    inverted_word_idx = dict((i + 1, c) for i, c in enumerate(vocab))
    
    for i in range(memory_size):
        word_idx['time{}'.format(i+1)] = 'time{}'.format(i+1)
    vocab_size = len(word_idx) + 1 
    sentence_size = max(query_size, sentence_size)
    sentence_size += 1 

    S, Q, A = vectorize_data(train, word_idx, sentence_size, memory_size)
    trainS, valS, trainQ, valQ, trainA, valA = train_test_split(S, Q, A, test_size=.1, random_state=FLAGS.random_state)
    testS, testQ, testA = vectorize_data(test, word_idx, sentence_size, memory_size)
    n_test, n_train, n_val = get_shapes(testS, trainS, valS)
    train_labels, test_labels, val_labels = get_labels(trainA, testA, valA)
    tf.set_random_seed(FLAGS.random_state)
    batch_size, batches = get_batches(n_train, FLAGS)

    with tf.Session() as sess:
        model = MemN2N(batch_size, vocab_size, sentence_size, memory_size, FLAGS.embedding_size, session=sess,
                    hops=FLAGS.hops, max_grad_norm=FLAGS.max_grad_norm)
        
        for t in range(1, FLAGS.epochs+1):
            lr = get_updated_learning_rate(t, FLAGS)
            np.random.shuffle(batches)
            total_cost = get_total_cost(batches, model, trainS, trainQ, trainA, lr)
            if t % FLAGS.evaluation_interval == 0:
                train_predictions = get_train_predictions(FLAGS, trainS, trainQ, n_train, batch_size, model)
                validation_predictions = model.predict(valS, valQ)
                training_accuracy = metrics.accuracy_score(np.array(train_predictions), train_labels)
                validation_accuracy = metrics.accuracy_score(validation_predictions, val_labels)

                print("******************************************")
                print('Epoch', t)
                print('Training Accuracy:', training_accuracy)
                print('Validation Accuracy:', validation_accuracy)
                print('Total Cost:', total_cost)
                print("******************************************")
                
        # Saving Model
        # input_dic = {"stories":model._stories, "queries":model._queries, "answers":model._answers, "learning_rate":model._lr}
        # output_dic = {model._logits}
        # tf.saved_model.simple_save(
        #     sess, "./saved_models/task_"+FLAGS.task_id, input_dic, output_dic
        # )

        test_predictions = model.predict(testS, testQ)
        testing_accuracy = metrics.accuracy_score(test_predictions, test_labels)
        print("Final Testing Accuracy:", testing_accuracy)
        print("")
        printTests(1,6,test, test_predictions, test_labels, inverted_word_idx)

if __name__ == "__main__" :
    print('Enter task id')
    id = int(input())
    main(id)