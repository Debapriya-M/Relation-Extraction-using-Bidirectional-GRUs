import tensorflow as tf
from tensorflow.keras import layers, models

from util import ID_TO_CLASS


class MyBasicAttentiveBiGRU(models.Model):

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int = 128, training: bool = False):
        super(MyBasicAttentiveBiGRU, self).__init__()

        self.num_classes = len(ID_TO_CLASS)

        self.decoder = layers.Dense(units=self.num_classes)
        self.omegas = tf.Variable(tf.random.normal((hidden_size*2, 1)))
        self.embeddings = tf.Variable(tf.random.normal((vocab_size, embed_dim)))

        ### TODO(Students) START
        # ...
    
        self.forward = layers.GRU(hidden_size, activation='tanh', use_bias=True, 
            return_sequences=True, recurrent_activation = 'sigmoid')
        self.backward = layers.GRU(hidden_size, activation='tanh', use_bias=True, 
            return_sequences=True, go_backwards = True, recurrent_activation = 'sigmoid')

        self.bidirectional = layers.Bidirectional(self.forward, backward_layer =self.backward, input_shape =(embed_dim, hidden_size))

        ### TODO(Students) END

    def attn(self, rnn_outputs):
        ### TODO(Students) START
        # ...
        M = tf.tanh(rnn_outputs)
        alpha = tf.nn.softmax(tf.tensordot(M,self.omegas, axes = 1), axis = 1)
        r = tf.multiply(rnn_outputs, alpha)
        output = tf.reduce_sum(r, axis = 1)
        ### TODO(Students) END

        return output

    def call(self, inputs, pos_inputs, training):
        word_embed = tf.nn.embedding_lookup(self.embeddings, inputs)
        pos_embed = tf.nn.embedding_lookup(self.embeddings, pos_inputs)

        ### TODO(Students) START
        # ...
        masked_val = tf.cast(inputs!=0, tf.float32)
        # H = self.bidirectional(word_embed, mask = masked_val)
        H = self.bidirectional(tf.concat([word_embed,pos_embed],axis=2), mask = masked_val)
        final_representation = tf.math.tanh(self.attn(H))
        logits = self.decoder(final_representation)

        ### TODO(Students) END

        return {'logits': logits}


class MyAdvancedModel(models.Model):

    def __init__(self, vocab_size: int, embed_dim: int, training: bool = False):
        super(MyAdvancedModel, self).__init__()
        ### TODO(Students) START
        # ...
        self.num_classes = len(ID_TO_CLASS)
        self.decoder = layers.Dense(units=self.num_classes)
        self.embeddings = tf.Variable(tf.random.normal((vocab_size, embed_dim)))
        self.conv1 = layers.Conv1D(filters=256, kernel_size=2, padding='same', activation='relu')
        self.max_pool1 = layers.GlobalMaxPool1D()

        self.conv2 = layers.Conv1D(filters=256, kernel_size=3, padding='same', activation='relu')
        self.max_pool2 = layers.GlobalMaxPool1D()

        self.conv3 = layers.Conv1D(filters=256, kernel_size=4, padding='same', activation='relu')
        self.max_pool3 = layers.GlobalMaxPool1D()

        ### TODO(Students END


    def call(self, inputs, pos_inputs, training):
        # raise NotImplementedError
        ### TODO(Students) START
        # ...
        word_embed = tf.nn.embedding_lookup(self.embeddings, inputs)
        pos_embed = tf.nn.embedding_lookup(self.embeddings, pos_inputs)

        sent_pos = tf.concat([word_embed, pos_embed], axis=2)
        # keep_prob = 0.5
        # sent_pos = tf.nn.dropout(sent_pos, keep_prob)
        
        conv_layer1 = self.conv1(sent_pos)
        pooling1 = self.max_pool1(conv_layer1)

        conv_layer2 = self.conv2(conv_layer1)
        pooling2 = self.max_pool2(conv_layer2)

        conv_layer3 = self.conv3(conv_layer2)
        pooling3 = self.max_pool3(conv_layer3)

        pool = tf.concat([pooling1, pooling2, pooling3], axis=1)
        # feature = tf.nn.dropout(pool, keep_prob)

        logits = self.decoder(pool)

        return {'logits': logits}
        ### TODO(Students END
