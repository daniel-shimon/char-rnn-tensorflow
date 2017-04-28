from __future__ import print_function

import sys
import tensorflow as tf

import argparse
import os
from six.moves import cPickle

from model import Model

from six import text_type


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save-dir', type=str, default=None,
                        help='model directory to store checkpointed models')
    parser.add_argument('--dataset', type=str, default=None,
                        help='single name to use under save directory')
    parser.add_argument('--output-file', type=str, default=None,
                        help='output sample to a specific file')
    parser.add_argument('-n', type=int, default=500,
                        help='number of characters to sample '
                             '(maximum when using start and stop chars)')
    parser.add_argument('--prime', type=text_type, default=u' ',
                        help='prime text')
    parser.add_argument('--whole', action='store_true',
                        help='prime the text with a start char and finish at the first end char '
                             '(or after n chars when n >= 0)')
    parser.add_argument('--sample', type=int, default=1,
                        help='0 to use max at each timestep, 1 to sample at '
                             'each timestep, 2 to sample on spaces')

    args = parser.parse_args()
    sample(args)


def sample(args):
    chars, model, vocab = load_data(args)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            output = model.sample(sess, chars, vocab, args.n, args.prime,
                                     args.sample, args.whole)

            if args.output_file is not None:
                dir_name = os.path.dirname(args.output_file)
                if dir_name != '':
                    os.makedirs(dir_name)
                with open(args.output_file, 'w') as output_file:
                    output_file.write(''.join(output))
            else:
                for char in output:
                    sys.stdout.write(char)
                sys.stdout.flush()


def load_data(args):
    if args.dataset:
        if not args.save_dir:
            args.save_dir = os.path.join('save', args.dataset)

    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f)

    model = Model(saved_args, training=False)

    return chars, model, vocab


if __name__ == '__main__':
    main()
