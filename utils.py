import codecs
import os
import random

from six.moves import cPickle
import numpy as np


class TextLoader:
    def __init__(self, name, shared_vocabulary, batch_size, seq_length):
        self.tensor = shared_vocabulary.tensors[name]
        self.split_mode = isinstance(self.tensor, list)
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.partition_size = batch_size * seq_length
        self.name = name

        self.num_batches = 0
        self.x_batches = None
        self.y_batches = None
        self.pointer = None

        self.create_batches()
        self.reset_batch_pointer()

    def create_batches(self):
        if self.split_mode:
            tensor = []
            indices = range(len(self.tensor))
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
        assert self.num_batches != 0, "Not enough data in %s. Make batch_size smaller." % self.name

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


class SharedVocabulary:
    def __init__(self, data_dir, names, encoding='utf-8'):
        self.chars = set()
        self.vocab = None
        self.vocab_size = 0
        self.data = dict()
        self.tensors = dict()
        self.encoding = encoding
        self.split_mode = False
        self.vocab_file = os.path.join(data_dir, "vocab.pkl")
        self.tensor_file = os.path.join(data_dir, "data.npy")

        if not (os.path.exists(self.vocab_file) and os.path.exists(self.tensor_file)):
            for name in names:
                input_file = os.path.join(data_dir, name + ".txt")
                input_folder = os.path.join(data_dir, name + "s")
                if os.path.exists(input_file):
                    print("reading text file")
                    self.read_file(name, input_file)
                elif os.path.exists(input_folder):
                    print("reading text files")
                    self.split_mode = True
                    self.read_folder(name, input_folder)
                else:
                    raise EnvironmentError('neither {} nor {} exist'.format(input_file, input_folder))

            self.process_data()
        else:
            with open(self.vocab_file, 'rb') as f:
                self.chars, self.vocab_size, self.vocab, self.split_mode = cPickle.load(f)
            with open(self.tensor_file, 'rb') as f:
                self.tensors = cPickle.load(f)

    def read_file(self, name, input_file):
        with codecs.open(input_file, "r", encoding=self.encoding) as f:
            data = f.read()
        self.chars.update(data)
        self.data[name] = data

    def read_folder(self, name, input_folder):
        data = []
        # load multiple files with start-of-text and end-of-text chars
        for filename in os.listdir(input_folder):
            with codecs.open(os.path.join(input_folder, filename),
                             "r", encoding=self.encoding) as f:
                data.append('\x02' + f.read() + '\x03')
            self.chars.update(data[-1])

        self.data[name] = data

    def process_data(self):
        self.chars = list(self.chars)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        with open(self.vocab_file, 'wb') as f:
            cPickle.dump((self.chars, self.vocab_size, self.vocab, self.split_mode), f)

        for name, data in self.data.items():
            if isinstance(data, list):
                # split mode:
                tensors = []
                for single_example in data:
                    tensors.append(np.array(list(map(self.vocab.get, single_example))))
                self.tensors[name] = tensors
            else:
                if self.split_mode:
                    data.insert(0, '\x02')
                    data.append('\x03')
                self.tensors[name] = np.array(list(map(self.vocab.get, data)))

        # save tensors
        with open(self.tensor_file, 'wb') as f:
            cPickle.dump(self.tensors, f)
