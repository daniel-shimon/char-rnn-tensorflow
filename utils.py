import codecs
import os
import collections
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
        self.clean_state_batches = []

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding

        input_file = os.path.join(data_dir, "input.txt")
        input_folder = os.path.join(data_dir, "inputs")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            if os.path.exists(input_file):
                print("reading text file")
                self.split_mode = False
                self.preprocess(input_file, vocab_file, tensor_file)
            elif os.path.exists(input_folder):
                print("reading text files")
                self.split_mode = True
                self.preprocess(input_folder, vocab_file, tensor_file)
            else:
                raise EnvironmentError('neither {} nor {} exist'.format(input_file, input_folder))

        else:
            print("loading preprocessed files")
            # return whether in split mode
            self.load_preprocessed(vocab_file, tensor_file)
        self.create_batches()
        self.reset_batch_pointer()

    def preprocess(self, data_path, vocab_file, tensor_file):
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
        with open(vocab_file, 'wb') as f:
            cPickle.dump((self.chars, self.vocab_size, self.vocab, self.split_mode), f)

        if self.split_mode:
            tensors = []
            for single_example in data:
                tensors.append(np.array(list(map(self.vocab.get, single_example))))
            self.tensor = np.array(tensors)
        else:
            self.tensor = np.array(list(map(self.vocab.get, data)))

        np.save(tensor_file, self.tensor)

    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.chars, self.vocab_size, self.vocab, self.split_mode = cPickle.load(f)

        self.tensor = np.load(tensor_file)

    def create_batches(self):
        if self.split_mode:
            self.clean_state_batches = []
            self.x_batches = []
            self.y_batches = []
            self.num_batches = 0

            # iterate over examples (files)
            for i in range(self.tensor.shape[0]):
                num_batches, x_batches, y_batches = self.tensor_to_batches(self.tensor[i])
                self.clean_state_batches.append(self.num_batches)
                self.num_batches += num_batches
                self.x_batches.extend(x_batches)
                self.y_batches.extend(y_batches)
        else:
            self.num_batches, self.x_batches, self.y_batches = self.tensor_to_batches(self.tensor)

    def tensor_to_batches(self, tensor):
        num_batches = int((tensor.size - 1) / self.batch_size)
        x_batches = []
        y_batches = []

        # When the data (tensor) is too small,
        # let's give them a better error message
        if num_batches == 0:
            if self.split_mode:
                extra_tensor = tensor
            else:
                raise AssertionError("Not enough data. Make batch_size smaller.")
        else:
            clipped_tensor = tensor[:num_batches * self.batch_size + 1]
            extra_tensor = tensor[num_batches * self.batch_size:]
            x_data = clipped_tensor[:-1]
            y_data = np.copy(clipped_tensor[1:])

            x_batches = np.split(x_data.reshape(self.batch_size, -1),
                                 num_batches, 1)
            y_batches = np.split(y_data.reshape(self.batch_size, -1),
                                 num_batches, 1)

        if extra_tensor.size > 1:
            num_batches += 1
            x_data = extra_tensor[:-1]
            y_data = np.copy(extra_tensor[1:])
            x_batches.append(x_data.reshape(-1, 1))
            y_batches.append(y_data.reshape(-1, 1))

        return num_batches, x_batches, y_batches

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0
