# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2021.  Mohamed Reda Bouadjenek, Deakin University                       +
#               Email:  reda.bouadjenek@deakin.edu.au                                    +
#                                                                                        +
#       Licensed under the Apache License, Version 2.0 (the "License");                  +
#       you may not use this file except in compliance with the License.                 +
#       You may obtain a copy of the License at:                                         +
#                                                                                        +
#       http://www.apache.org/licenses/LICENSE-2.0                                       +
#                                                                                        +
#       Unless required by applicable law or agreed to in writing, software              +
#       distributed under the License is distributed on an "AS IS" BASIS,                +
#       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.         +
#       See the License for the specific language governing permissions and              +
#       limitations under the License.                                                   +
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\
import tensorflow as tf
from tensorflow import keras
from transformers import TFBertModel
from tensorflow.keras import layers


def create_model(max_len, num_labels):
    ## BERT encoder
    encoder = TFBertModel.from_pretrained("bert-base-uncased")

    ## Model
    input_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
    attention_mask = layers.Input(shape=(max_len,), dtype=tf.int32)
    embedding = encoder(
        input_ids, attention_mask=attention_mask
    )['pooler_output']
    # dense = layers.Dropout(0.99)(embedding)

    dense = layers.Dense(1024, activation='relu')(embedding)
    dense = layers.Dropout(.4)(dense)
    out = layers.Dense(num_labels, activation='softmax')(dense)

    model = keras.Model(
        inputs=[input_ids, attention_mask],
        outputs=out,
    )
    loss = keras.losses.SparseCategoricalCrossentropy()
    optimizer = keras.optimizers.Adam(learning_rate=5e-5)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model


def get_model(max_len, num_labels, tpu=True):
    if tpu:
        # Create distribution strategy
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)

        # Create model
        with strategy.scope():
            model = create_model(max_len, num_labels)
    else:
        model = create_model(max_len, num_labels)
    return model


if __name__ == '__main__':
    print('')
