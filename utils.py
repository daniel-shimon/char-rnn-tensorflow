import codecs
import os
import random

from six.moves import cPickle
import numpy as np


class TextLoader:
    def __init__(self, data_dir, batch_size, seq_length, encoding='utf-8'):
        self.chars = 0
        self.vocab = None
        self.vocab_size = 0
        self.tensor = None
        self.num_batches = 0
        self.x_batches = None
        self.y_batches = None
        self.pointer = None
        self.split_mode = None

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.partition_size = batch_size * seq_length
        self.encoding = encoding

        input_file = os.path.join(data_dir, "input.txt")
        input_folder = os.path.join(data_dir, "inputs")
        self.vocab_file = os.path.join(data_dir, "vocab.pkl")
        self.input_tensor_file = os.path.join(data_dir, "data.npy")

        if not (os.path.exists(self.vocab_file) and os.path.exists(self.input_tensor_file)):
            if os.path.exists(input_file):
                print("reading text file")
                self.split_mode = False
                self.preprocess(input_file)
            elif os.path.exists(input_folder):
                print("reading text files")
                self.split_mode = True
                self.preprocess(input_folder)
            else:
                raise EnvironmentError('neither {} nor {} exist'.format(input_file, input_folder))

        else:
            print("loading preprocessed files")
            self.load_preprocessed()
        self.create_batches()
        self.reset_batch_pointer()

    def preprocess(self, data_path):
        if self.split_mode:
            data = []
            self.chars = set()
            # load multiple files with start-of-text and end-of-text chars
            for filename in os.listdir(data_path):
                with codecs.open(os.path.join(data_path, filename),
                                 "r", encoding=self.encoding) as f:
                    data.append('\x02' + f.read() + '\x03')
                self.chars.update(data[-1])
        else:
            with codecs.open(data_path, "r", encoding=self.encoding) as f:
                data = f.read()
            self.chars = set(data)

        self.chars = list(self.chars)

        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        with open(self.vocab_file, 'wb') as f:
            cPickle.dump((self.chars, self.vocab_size, self.vocab, self.split_mode), f)

        if self.split_mode:
            tensors = []
            for single_example in data:
                tensors.append(np.array(list(map(self.vocab.get, single_example))))
            self.tensor = np.array(tensors)
        else:
            self.tensor = np.array(list(map(self.vocab.get, data)))

        np.save(self.input_tensor_file, self.tensor)

    def load_preprocessed(self):
        with open(self.vocab_file, 'rb') as f:
            self.chars, self.vocab_size, self.vocab, self.split_mode = cPickle.load(f)

        self.tensor = np.load(self.input_tensor_file)

    def create_batches(self):
        if self.split_mode:
            tensor = []
            indices = range(self.tensor.shape[0])
            random.shuffle(indices)

            for i in indices:
                tensor.extend(self.tensor[i])

            tensor = np.array(tensor)
        else:
            tensor = self.tensor

        # create batches
        self.num_batches = int((tensor.size - 1) / self.partition_size)

        # When the data (tensor) is too small,
        # let's give them a better error message
        assert self.num_batches != 0, "Not enough data. Make batch_size smaller."

        clipped_tensor = tensor[:self.num_batches * self.partition_size + 1]
        x_data = clipped_tensor[:-1]
        y_data = np.copy(clipped_tensor[1:])

        self.x_batches = np.split(x_data.reshape(self.batch_size, -1),
                                  self.num_batches, 1)
        self.y_batches = np.split(y_data.reshape(self.batch_size, -1),
                                  self.num_batches, 1)

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0
