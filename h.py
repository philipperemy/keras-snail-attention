"""
    def forward(self, minibatch):
        minibatch = minibatch.permute(0,2,1)
        keys = self.key_layer(minibatch)
        queries = self.query_layer(minibatch)
        values = self.value_layer(minibatch)
        logits = torch.bmm(queries, keys.transpose(2,1))
        mask = logits.data.new(logits.size(1), logits.size(2)).fill_(1).byte()
        mask = torch.triu(mask, 1)
        mask = mask.unsqueeze(0).expand_as(logits)
        logits.data.masked_fill_(mask, float('-inf'))
        probs = F.softmax(logits / self.sqrt_k, dim=2)
        read = torch.bmm(probs, values)
        return torch.cat([minibatch, read], dim=2).permute(0,2,1)
"""
import math
from argparse import ArgumentParser

import keras.backend as K
import numpy as np
from keras import Input, Model
from keras.datasets import imdb
from keras.layers import Dense, Embedding, Flatten
from keras.layers import LSTM
from keras.layers import Layer, Softmax
from keras.preprocessing import sequence

max_features = 20000
# cut texts after this number of words (among top max_features most common words)
maxlen = 80
batch_size = 32


class AttentionBlock(Layer):

    def __init__(self, dims, k_size, v_size, seq_len=None, **kwargs):
        self.k_size = k_size
        self.seq_len = seq_len
        self.v_size = v_size
        self.dims = dims
        self.sqrt_k = math.sqrt(k_size)
        self.keys_fc = None
        self.queries_fc = None
        self.values_fc = None
        super(AttentionBlock, self).__init__(**kwargs)

    def build(self, input_shape):
        # https://stackoverflow.com/questions/54194724/how-to-use-keras-layers-in-custom-keras-layer
        self.keys_fc = Dense(self.k_size)
        self.keys_fc.build((None, self.dims))
        self._trainable_weights.extend(self.keys_fc.trainable_weights)

        self.queries_fc = Dense(self.k_size)
        self.queries_fc.build((None, self.dims))
        self._trainable_weights.extend(self.queries_fc.trainable_weights)

        self.values_fc = Dense(self.v_size)
        self.values_fc.build((None, self.dims))
        self._trainable_weights.extend(self.values_fc.trainable_weights)

    def call(self, inputs, **kwargs):
        # check that the implementation matches exactly py torch.
        keys = self.keys_fc(inputs)
        queries = self.queries_fc(inputs)
        values = self.values_fc(inputs)
        logits = K.batch_dot(queries, K.permute_dimensions(keys, (0, 2, 1)))
        mask = K.ones_like(logits) * np.triu((-np.inf) * np.ones(logits.shape.as_list()[1:]), k=1)
        logits = mask + logits
        probs = Softmax(axis=-1)(logits / self.sqrt_k)
        read = K.batch_dot(probs, values)
        output = K.concatenate([inputs, read], axis=-1)
        return output

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] += self.v_size
        return tuple(output_shape)


def get_script_arguments():
    args = ArgumentParser()
    args.add_argument('--attention', action='store_true')
    args.add_argument('--units', default=24, type=int)
    return args.parse_args()


def main():
    args = get_script_arguments()
    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    print('Build model...')
    i = Input(shape=(maxlen,))
    x = Embedding(max_features, args.units)(i)
    x = LSTM(args.units, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(x)
    if args.attention:
        x = AttentionBlock(args.units, 24, 12)(x)
        x = Dense(args.units)(x)
    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=[i], outputs=[x])

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    print('Train...')
    print(x_train.shape)
    print(y_train.shape)
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=15, validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test,
                                batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)


if __name__ == '__main__':
    main()
