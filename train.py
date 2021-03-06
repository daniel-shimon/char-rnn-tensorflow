from __future__ import print_function
import tensorflow as tf

import argparse
import time
import os
from six.moves import cPickle

from utils import TextLoader, SharedVocabulary
from model import Model


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # environment:
    parser.add_argument('--dataset', type=str, default=None,
                        help='single name to use under directories (data, save and log)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='data directory containing input.txt')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='directory to store checkpointed models')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='directory to store tensorboard logs')
    parser.add_argument('--init-from', type=str, default=None,
                        help='continue training from saved model at this directory')

    # neural network:
    parser.add_argument('--rnn-size', type=int, default=256,
                        help='size of RNN hidden state')
    parser.add_argument('--num-layers', type=int, default=1,
                        help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, lstm, or nas')

    # training:
    parser.add_argument('--batch-size', type=int, default=50,
                        help='minibatch size')
    parser.add_argument('--seq-length', type=int, default=50,
                        help='RNN sequence length')
    parser.add_argument('--num-epochs', type=int, default=50,
                        help='number of epochs')
    parser.add_argument('--grad-clip', type=float, default=5.,
                        help='clip gradients at this value')
    parser.add_argument('--learning-rate', type=float, default=0.002,
                        help='learning rate')
    parser.add_argument('--decay-rate', type=float, default=0.97,
                        help='decay rate for rmsprop')
    parser.add_argument('--output-keep-prob', type=float, default=1.0,
                        help='probability of keeping weights in the hidden layer')
    parser.add_argument('--input-keep-prob', type=float, default=1.0,
                        help='probability of keeping weights in the input layer')

    # extra:
    parser.add_argument('--save-every', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--validation-every', type=int, default=1,
                        help='validation frequency (epochs)')
    args = parser.parse_args()
    train(args)


def train(args):
    sort_environment(args)

    data_loader, test_loader, split_mode = load_data(args)
    ckpt, model = load_model(args)
    initial_iteration = 0

    with tf.Session() as sess:
        # instrument for tensorboard
        train_writer = tf.summary.FileWriter(
            os.path.join(args.log_dir, time.strftime("%d_%m_%y__%H_%M_%S__train")))
        test_writer = tf.summary.FileWriter(
            os.path.join(args.log_dir, time.strftime("%d_%m_%y__%H_%M_%S__test")))
        train_writer.add_graph(sess.graph)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())

        # restore model
        if args.init_from is not None:
            if os.path.exists(args.init_from):
                print('initiating model from ' + args.init_from)
                saver.restore(sess, ckpt.model_checkpoint_path)
                with open(os.path.join(args.save_dir, 'step.info'), 'r') as f:
                    initial_iteration = int(f.read())

        for e in range(args.num_epochs):
            if e % args.validation_every == 0:
                if split_mode and e > 0:
                    test_loader.create_batches()
                test_loader.reset_batch_pointer()
                state = sess.run(model.initial_state)

                for b in range(test_loader.num_batches):
                    current_iteration = e * data_loader.num_batches + b
                    start = time.time()

                    x, y = test_loader.next_batch()
                    test_loss = batch(current_iteration, initial_iteration, model,
                                      sess, state, test_writer, x, y)

                    end = time.time()
                    print("test {}/{} (epoch {}), loss = {:.3f}, time/batch = {:.3f}"
                          .format(b + 1,
                                  test_loader.num_batches,
                                  e, test_loss, end - start))

            if split_mode and e > 0:
                data_loader.create_batches()
            data_loader.reset_batch_pointer()
            state = sess.run(model.initial_state)
            sess.run(tf.assign(model.lr,
                               args.learning_rate * (args.decay_rate ** e)))

            for b in range(data_loader.num_batches):
                current_iteration = e * data_loader.num_batches + b
                start = time.time()

                x, y = data_loader.next_batch()
                train_loss = batch(current_iteration, initial_iteration, model,
                                   sess, state, train_writer, x, y)

                end = time.time()
                print("{}/{} (epoch {}), loss = {:.3f}, time/batch = {:.3f}"
                      .format(current_iteration + 1,
                              args.num_epochs * data_loader.num_batches,
                              e, train_loss, end - start))

                if current_iteration % args.save_every == 0 \
                        or (e == args.num_epochs - 1 and
                                    b == data_loader.num_batches - 1):
                    # save for the last result
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path,
                               global_step=initial_iteration + current_iteration)
                    with open(os.path.join(args.save_dir, 'step.info'), 'w') as f:
                        f.write(str(initial_iteration + current_iteration + 1))
                    print("model saved to {}".format(args.save_dir))


def batch(current_iteration, initial_iteration, model, sess, state, writer, x, y):
    feed = {model.input_data: x, model.targets: y}

    for i, (c, h) in enumerate(model.initial_state):
        feed[c] = state[i].c
        feed[h] = state[i].h

    summary_results, train_loss, state, _ = \
        sess.run([model.summary, model.cost, model.final_state, model.train_op], feed)

    # instrument for tensorboard
    writer.add_summary(summary_results, initial_iteration + current_iteration)
    return train_loss


def sort_environment(args):
    if args.dataset:
        if not args.data_dir:
            args.data_dir = os.path.join('data', args.dataset)
        if not args.save_dir:
            args.save_dir = os.path.join('save', args.dataset)
        if not args.log_dir:
            args.log_dir = os.path.join('logs', args.dataset)
        if not args.init_from:
            args.init_from = args.save_dir

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)


def load_model(args):
    ckpt = None

    # check compatibility if training is continued from previously saved model
    if args.init_from is not None:
        # check if all necessary files exist
        try:
            assert os.path.isdir(args.init_from), " %s must be a a path" % args.init_from
            assert os.path.isfile(
                os.path.join(args.init_from, "config.pkl")), \
                "config.pkl file does not exist in path %s" % args.init_from
            assert os.path.isfile(os.path.join(args.init_from,
                                               "chars_vocab.pkl")), \
                "chars_vocab.pkl file does not exist in path %s" % args.init_from
            assert os.path.isfile(os.path.join(args.init_from,
                                               "step.info")), \
                "step.info file does not exist in path %s" % args.init_from
            ckpt = tf.train.get_checkpoint_state(args.init_from)
            assert ckpt, "No checkpoint found"
            assert ckpt.model_checkpoint_path, "No model path found in checkpoint"

            # open old config and check if models are compatible
            with open(os.path.join(args.init_from, 'config.pkl'), 'rb') as f:
                saved_model_args = cPickle.load(f)
            need_be_same = ["model", "rnn_size", "num_layers"]
            for key in need_be_same:
                saved_value = vars(saved_model_args)[key]
                if vars(args)[key] is None:
                    setattr(args, key, saved_value)
                else:
                    assert saved_value == vars(args)[key], \
                        "Command line argument and saved model disagree on '%s' " % key

        except AssertionError as e:
            if args.dataset:
                print('model from ' + args.init_from + ' will not be used:', str(e))
                args.init_from = None
            else:
                raise e

    assert args.rnn_size is not None, 'missing rnn size'
    assert args.num_layers is not None, 'missing rnn size'

    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(args, f)

    model = Model(args)

    return ckpt, model


def load_data(args):
    shared_vocab = SharedVocabulary(args.data_dir, ['input', 'test'])
    args.vocab_size = shared_vocab.vocab_size
    data_loader = TextLoader('input', shared_vocab, args.batch_size, args.seq_length)
    test_loader = TextLoader('test', shared_vocab, args.batch_size, args.seq_length)

    if args.init_from is not None:
        # check if all necessary files exist
        try:
            assert os.path.isdir(args.init_from), " %s must be a a path" % args.init_from
            assert os.path.isfile(
                os.path.join(args.init_from, "config.pkl")), \
                "config.pkl file does not exist in path %s" % args.init_from
            assert os.path.isfile(os.path.join(args.init_from,
                                               "chars_vocab.pkl")), \
                "chars_vocab.pkl file does not exist in path %s" % args.init_from
            assert os.path.isfile(os.path.join(args.init_from,
                                               "step.info")), \
                "step.info file does not exist in path %s" % args.init_from
            ckpt = tf.train.get_checkpoint_state(args.init_from)
            assert ckpt, "No checkpoint found"
            assert ckpt.model_checkpoint_path, "No model path found in checkpoint"

            # open old config and check if models are compatible
            with open(os.path.join(args.init_from, 'config.pkl'), 'rb') as f:
                saved_model_args = cPickle.load(f)
            need_be_same = ["model", "rnn_size", "num_layers"]
            for key in need_be_same:
                saved_value = vars(saved_model_args)[key]
                if vars(args)[key] is None:
                    setattr(args, key, saved_value)
                else:
                    assert saved_value == vars(args)[key], \
                        "Command line argument and saved model disagree on '%s' " % key

            # open saved vocab/dict and check if vocabs/dicts are compatible
            with open(os.path.join(args.init_from, 'chars_vocab.pkl'), 'rb') as f:
                saved_chars, saved_vocab, saved_split_mode = cPickle.load(f)
            assert saved_chars == shared_vocab.chars, "Data and loaded model disagree on character set!"
            assert saved_vocab == shared_vocab.vocab, "Data and loaded model disagree on dictionary mappings!"

        except AssertionError as e:
            if args.dataset:
                print('model from ' + args.init_from + ' will not be used:', str(e))
                args.init_from = None
            else:
                raise e

    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'wb') as f:
        cPickle.dump((shared_vocab.chars, shared_vocab.vocab, shared_vocab.split_mode), f)

    return data_loader, test_loader, shared_vocab.split_mode


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main()
