import spektral
import tensorflow as tf

from attention import Attention


class perCLTV(tf.keras.Model):
    def __init__(self,
                 timestep=10,
                 behavior_num=100 + 1,
                 behavior_emb_dim=16,
                 behavior_maxlen=64,
                 behavior_dim=32,
                 network_dim=32,
                 dropout=0.5):
        super(perCLTV, self).__init__()

        self.timestep = timestep
        self.dropout = dropout
        self.behavior_num = behavior_num
        self.behavior_emb_dim = behavior_emb_dim
        self.behavior_maxlen = behavior_maxlen
        self.behavior_dim = behavior_dim
        self.network_dim = network_dim

        self.embedding = tf.keras.layers.Embedding(input_dim=self.behavior_num,
                                                   output_dim=self.behavior_emb_dim,
                                                   mask_zero=True)

        self.individual_behavior_nets = [tf.keras.Sequential(
            name='individual_behavior_net_'+str(x),
            layers=[self.embedding,
                    tf.keras.layers.Lambda(lambda x: tf.reshape(
                        x, (-1, self.behavior_maxlen, self.behavior_emb_dim))),
                    tf.keras.layers.LSTM(units=self.behavior_dim,
                                         return_sequences=False),
                    tf.keras.layers.Lambda(lambda x: tf.reshape(
                        x, (-1, self.timestep, self.behavior_dim))),
                    tf.keras.layers.LSTM(units=self.behavior_dim,
                                         return_sequences=True),
                    Attention(units=self.behavior_dim, score='luong'),
                    tf.keras.layers.Dense(units=self.behavior_dim,
                                          activation='relu',
                                          use_bias=False)]) for x in range(3)]

        self.social_behavior_net = tf.keras.Sequential(
            name='social_behavior_net',
            layers=[spektral.layers.GATConv(channels=self.network_dim,
                                            activation='relu'),
                    tf.keras.layers.Dropout(rate=self.dropout),
                    tf.keras.layers.Dense(units=self.network_dim,
                                          activation='relu')])
        self.hban = tf.keras.Sequential(name='hierarchical_behavioral_attention_net',
                                        layers=[tf.keras.layers.Concatenate(axis=-1),
                                                tf.keras.layers.Conv1D(
                                            filters=self.behavior_dim, kernel_size=1),
                                            tf.keras.layers.GlobalAveragePooling1D()])
        self.h_churn = tf.keras.layers.Dense(
            units=self.network_dim, activation='relu', name='h_churn')
        self.h_pay = tf.keras.layers.Dense(
            units=self.network_dim, activation='relu', name='h_pay')
        self.gate = tf.keras.layers.Dense(
            units=self.network_dim, activation='sigmoid', name='gate')
        self.output1 = tf.keras.layers.Dense(
            units=1, activation='sigmoid', name='output1')
        self.output2 = tf.keras.layers.Dense(
            units=1, activation=None, name='output2')

    def call(self, inputs):
        B, C, P, A = inputs
        X_B = tf.expand_dims(self.individual_behavior_nets[0](B), axis=-1)
        X_C = tf.expand_dims(self.individual_behavior_nets[1](C), axis=-1)
        X_P = tf.expand_dims(self.individual_behavior_nets[2](P), axis=-1)
        X = self.hban([X_B, X_C, X_P])
        O = self.social_behavior_net([X, A])
        h_churn = self.h_churn(O)
        h_pay = self.h_pay(O)
        gate = self.gate(O)
        output1 = self.output1(h_churn)
        output2 = self.output2(tf.keras.layers.Concatenate()(
            [tf.keras.layers.Multiply()([gate, h_churn]), h_pay]))
        return output1, output2
