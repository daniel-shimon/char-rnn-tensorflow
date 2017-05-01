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
                        help='number of characters to sample')
    parser.add_argument('--prime', type=text_type, default=u' ',
                        help='prime text')
    parser.add_argument('--temp', type=float, default=0.5,
                        help='temperature (0 < temp <= 1)')
    parser.add_argument('--sample', type=int, default=1,
                        help='0 to use max at each timestep, 1 to sample at '
                             'each timestep, 2 to sample on spaces')

    args = parser.parse_args()
    sample(args)


def sample(args):
    chars, model, vocab, split_mode = load_data(args)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            output = model.sample(sess, chars, vocab, args.n, args.prime,
                                  args.sample, split_mode, args.temp)

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
        chars, vocab, split_mode = cPickle.load(f)

    model = Model(saved_args, training=False)

    return chars, model, vocab, split_mode


if __name__ == '__main__':
    main()
