import numpy as np
import pytest
import tensorflow as tf

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def load_data():
    # MNISTデータセットのロード
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return x_train, y_train, x_test, y_test

def create_and_train_model(x_train, y_train):
    # データの前処理
    x_train = x_train.reshape(60000, 784).astype('float32') / 255
    y_train = tf.keras.utils.to_categorical(y_train, 10)

    # モデルの作成
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # モデルのコンパイル
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # モデルの訓練
    model.fit(x_train, y_train, epochs=5, batch_size=32)
    return model

def test_load_data():
    x_train, y_train, x_test, y_test = load_data()
    # データの形状とタイプをチェックする
    assert x_train.shape == (60000, 28, 28)
    assert y_train.shape == (60000,)
    assert x_test.shape == (10000, 28, 28)
    assert y_test.shape == (10000,)
    assert x_train.dtype == np.uint8
    assert y_train.dtype == np.uint8

def test_create_and_train_model():
    x_train, y_train, _, _ = load_data()
    model = create_and_train_model(x_train, y_train)
    # モデルが正しく訓練されていることをチェックする
    assert len(model.layers) == 2
    assert model.layers[0].output_shape == (None, 128)
