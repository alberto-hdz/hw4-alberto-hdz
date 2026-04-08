#!/usr/bin/env python3
# HW4 - Transfer Learning on CIFAR-10
# Alberto Hernandez

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
import time

# ── Load and preprocess CIFAR-10 ──────────────────────────────────────────────
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# CIFAR-10 is 32x32; most Keras Applications expect at least 75x75.
# We resize to 96x96 to keep things manageable but compatible.
TARGET_SIZE = 96

def preprocess(x, y):
    x = tf.cast(x, tf.float32)
    x = tf.image.resize(x, (TARGET_SIZE, TARGET_SIZE))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)  # scales to [-1, 1]
    return x, y

BATCH_SIZE = 64
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.shuffle(50000).map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds  = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_ds  = test_ds.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

NUM_CLASSES = 10
EPOCHS = 20

def plot_history(history, tag):
    m = history.history
    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1)
    plt.plot(history.epoch, m['loss'],     label='train loss')
    plt.plot(history.epoch, m['val_loss'], label='val loss')
    plt.legend(); plt.ylabel('Loss'); plt.xlabel('Epoch')
    plt.subplot(2, 1, 2)
    plt.plot(history.epoch, m['accuracy'],     label='train acc')
    plt.plot(history.epoch, m['val_accuracy'], label='val acc')
    plt.legend(); plt.ylabel('Accuracy'); plt.xlabel('Epoch')
    plt.suptitle(tag)
    plt.tight_layout()
    plt.savefig(f'curves_{tag}.png')
    plt.close()
    print(f"Saved curves_{tag}.png")

# ── Q9: Transfer learning from MobileNetV2 pretrained on ImageNet ─────────────
print("\n=== Q9: Transfer Learning (MobileNetV2 pretrained on ImageNet) ===")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(TARGET_SIZE, TARGET_SIZE, 3),
    include_top=False,       # drop the ImageNet classification head
    weights='imagenet'
)
base_model.trainable = False  # freeze pretrained weights initially

tl_model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation='softmax'),
])

tl_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
tl_model.summary()

t0 = time.time()
history_tl = tl_model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS)
tl_time = time.time() - t0
print(f"Transfer learning training time: {tl_time:.1f}s")
plot_history(history_tl, 'transfer_learning')

_, tl_test_acc = tl_model.evaluate(test_ds, verbose=0)
print(f"Transfer learning test accuracy: {tl_test_acc:.1%}")

# ── Q10: Train same architecture from scratch (random weights) ────────────────
print("\n=== Q10: Train from Scratch (MobileNetV2, random weights) ===")

base_scratch = tf.keras.applications.MobileNetV2(
    input_shape=(TARGET_SIZE, TARGET_SIZE, 3),
    include_top=False,
    weights=None        # random initialization
)
base_scratch.trainable = True  # train all weights from scratch

scratch_model = models.Sequential([
    base_scratch,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation='softmax'),
])

scratch_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

t0 = time.time()
history_scratch = scratch_model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS)
scratch_time = time.time() - t0
print(f"From-scratch training time: {scratch_time:.1f}s")
plot_history(history_scratch, 'from_scratch')

_, scratch_test_acc = scratch_model.evaluate(test_ds, verbose=0)
print(f"From-scratch test accuracy: {scratch_test_acc:.1%}")

# ── Summary comparison ────────────────────────────────────────────────────────
print("\n=== Comparison Summary ===")
print(f"  Transfer learning:  test acc = {tl_test_acc:.1%},  time = {tl_time:.1f}s")
print(f"  From scratch:       test acc = {scratch_test_acc:.1%},  time = {scratch_time:.1f}s")

final_tl_train_acc     = history_tl.history['accuracy'][-1]
final_tl_val_acc       = history_tl.history['val_accuracy'][-1]
final_scratch_train_acc = history_scratch.history['accuracy'][-1]
final_scratch_val_acc   = history_scratch.history['val_accuracy'][-1]

print(f"\n  TL     — final train acc: {final_tl_train_acc:.1%}, final val acc: {final_tl_val_acc:.1%}")
print(f"  Scratch — final train acc: {final_scratch_train_acc:.1%}, final val acc: {final_scratch_val_acc:.1%}")
