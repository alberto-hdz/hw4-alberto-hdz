#!/usr/bin/env python3
# HW4 - Speech Keyword Detection and Regularization
# Alberto Hernandez

import os, glob, pathlib, time, random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op as frontend_op

# ── Reproducibility ───────────────────────────────────────────────────────────
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)

# ── Audio settings ────────────────────────────────────────────────────────────
i16min = -2**15
i16max = 2**15 - 1
fsamp = 16000
wave_length_ms = 1000
wave_length_samps = int(wave_length_ms * fsamp / 1000)
window_size_ms = 64
window_step_ms = 48
num_filters = 32
batch_size = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE

# ── Keywords and labels ───────────────────────────────────────────────────────
commands = ['yes', 'no']
silence_str = "_silence"
unknown_str = "_unknown"
label_list = [silence_str, unknown_str] + commands
num_labels = len(label_list)
print('label_list:', label_list)

# ── Dataset location ──────────────────────────────────────────────────────────
data_dir = pathlib.Path('/home/albertohdz/homework/hw4-alberto-hdz/data/mini_speech_commands_extracted/mini_speech_commands')
print(f"Using dataset at: {data_dir}")

# ── Collect file paths ────────────────────────────────────────────────────────
filenames = glob.glob(os.path.join(str(data_dir), '*', '*.wav'))
random.shuffle(filenames)
num_samples = len(filenames)
print(f"Total wav files found: {num_samples}")

if num_samples == 0:
    raise RuntimeError(f"No wav files found under {data_dir}. Check the path.")

num_train = int(0.8 * num_samples)
num_val   = int(0.1 * num_samples)
num_test  = num_samples - num_train - num_val
train_files = filenames[:num_train]
val_files   = filenames[num_train: num_train + num_val]
test_files  = filenames[-num_test:]
print(f"Train/Val/Test: {len(train_files)}/{len(val_files)}/{len(test_files)}")

# ── Audio helpers ─────────────────────────────────────────────────────────────
def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)

def get_label_from_path(file_path):
    parts = tf.strings.split(file_path, '/')
    word = parts[-2]
    in_set = tf.reduce_any(tf.equal(word, label_list))
    return tf.cond(in_set, lambda: word, lambda: tf.constant(unknown_str))

def get_waveform_and_label_from_file(file_path):
    label = get_label_from_path(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label

# ── Spectrogram ───────────────────────────────────────────────────────────────
def get_spectrogram_float(waveform):
    waveform_int = tf.cast(0.5 * waveform * (i16max - i16min), tf.int16)
    zero_padding = tf.zeros([wave_length_samps] - tf.shape(waveform_int), dtype=tf.int16)
    equal_length = tf.concat([waveform_int, zero_padding], 0)
    return frontend_op.audio_microfrontend(
        equal_length, sample_rate=fsamp, num_channels=num_filters,
        window_size=window_size_ms, window_step=window_step_ms
    )

def create_silence_dataset(num_waves, samples_per_wave, rms_noise_range=(0.01, 0.2)):
    rng = np.random.default_rng()
    rms_levels = rng.uniform(*rms_noise_range, size=num_waves)
    rand_waves = np.zeros((num_waves, samples_per_wave), dtype=np.float32)
    for i in range(num_waves):
        rand_waves[i, :] = rms_levels[i] * rng.standard_normal(samples_per_wave)
    labels = [silence_str] * num_waves
    return tf.data.Dataset.from_tensor_slices((rand_waves, labels))

def copy_with_noise(ds_input, rms_level=0.25):
    rng_gen = tf.random.Generator.from_seed(1234)
    wave_shape = tf.constant((wave_length_samps,))
    def add_noise(waveform, label):
        noise = rms_level * rng_gen.normal(shape=wave_shape)
        zero_padding = tf.zeros([wave_length_samps] - tf.shape(waveform), dtype=tf.float32)
        waveform = tf.concat([waveform, zero_padding], 0)
        return waveform + noise, label
    return ds_input.map(add_noise)

def wavds2specds(waveform_ds):
    wav, _ = next(waveform_ds.as_numpy_iterator())
    one_spec = get_spectrogram_float(tf.constant(wav, dtype=tf.float32))
    one_spec = tf.expand_dims(tf.expand_dims(one_spec, 0), -1)

    num_waves = sum(1 for _ in waveform_ds)
    spec_shape = (num_waves,) + tuple(one_spec.shape[1:])
    spec_grams = np.zeros(spec_shape, dtype=np.float32)
    labels_out = np.zeros(num_waves, dtype=np.int32)

    for idx, (wav, label) in enumerate(waveform_ds):
        if idx % 500 == 0:
            print(f"\r  {idx}/{num_waves} processed", end='', flush=True)
        wav_np = wav.numpy() if hasattr(wav, 'numpy') else np.array(wav)
        spec = get_spectrogram_float(tf.constant(wav_np, dtype=tf.float32))
        spec_grams[idx] = tf.expand_dims(tf.expand_dims(spec, 0), -1).numpy()
        lbl_bytes = label.numpy()
        lbl_str = lbl_bytes.decode('utf8') if isinstance(lbl_bytes, bytes) else str(lbl_bytes)
        labels_out[idx] = int(np.argmax(lbl_str == np.array(label_list)))
    print()
    return tf.data.Dataset.from_tensor_slices((spec_grams, labels_out))

def is_batched(ds):
    try:
        ds.unbatch()
        return True
    except Exception:
        return False

def preprocess_dataset(files, num_silent=None, noisy_reps=None,
                       limit_cmd0=None, limit_cmd1=None):
    if limit_cmd0 is not None or limit_cmd1 is not None:
        filtered, c0, c1 = [], 0, 0
        for f in files:
            word = f.split('/')[-2]
            if word == commands[0]:
                if limit_cmd0 is not None and c0 >= limit_cmd0:
                    continue
                c0 += 1
            elif word == commands[1]:
                if limit_cmd1 is not None and c1 >= limit_cmd1:
                    continue
                c1 += 1
            filtered.append(f)
        files = filtered
        print(f"  Kept {c0} of '{commands[0]}', {c1} of '{commands[1]}'")

    if num_silent is None:
        num_silent = int(0.2 * len(files)) + 1

    print(f"Processing {len(files)} files + {num_silent} silence samples")

    files_ds = tf.data.Dataset.from_tensor_slices(files)
    waveform_ds = files_ds.map(get_waveform_and_label_from_file,
                               num_parallel_calls=AUTOTUNE)

    if noisy_reps is not None:
        ds_cmds = waveform_ds.filter(
            lambda w, l: tf.reduce_any(tf.equal(l, commands))
        )
        for level in noisy_reps:
            waveform_ds = waveform_ds.concatenate(copy_with_noise(ds_cmds, rms_level=level))

    file_spec_ds = wavds2specds(waveform_ds)

    if num_silent > 0:
        silent_wav_ds = create_silence_dataset(num_silent, wave_length_samps)
        silent_spec_ds = wavds2specds(silent_wav_ds)
        return file_spec_ds.concatenate(silent_spec_ds)

    return file_spec_ds

def make_batched_ds(ds, approx_len):
    ds = ds.shuffle(max(int(approx_len * 1.2), 100))
    if not is_batched(ds):
        ds = ds.batch(batch_size)
    return ds.cache().prefetch(AUTOTUNE)

# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_model(model, test_ds, tag="model"):
    was_batched = is_batched(test_ds)
    if was_batched:
        test_ds = test_ds.unbatch()

    test_audio, test_labels = [], []
    for audio, label in test_ds:
        test_audio.append(audio.numpy())
        test_labels.append(label.numpy())
    test_audio  = np.array(test_audio)
    test_labels = np.array(test_labels)

    y_pred = np.argmax(model.predict(test_audio, verbose=0), axis=1)
    y_true = test_labels
    acc = np.mean(y_pred == y_true)
    print(f"[{tag}] Test accuracy: {acc:.1%}")

    cm = tf.math.confusion_matrix(y_true, y_pred).numpy()
    total = cm.sum()
    tpr = np.full(len(label_list), np.nan)
    fpr = np.full(len(label_list), np.nan)
    for i in range(len(label_list)):
        row_sum = cm[i, :].sum()
        col_sum = cm[:, i].sum()
        tpr[i] = cm[i, i] / row_sum if row_sum > 0 else np.nan
        denom = total - row_sum
        fpr[i] = (col_sum - cm[i, i]) / denom if denom > 0 else np.nan
        print(f"  TPR/FPR '{label_list[i]:10}': {tpr[i]:.3f} / {fpr[i]:.3f}")

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, xticklabels=label_list, yticklabels=label_list,
                annot=True, fmt='g')
    plt.gca().invert_yaxis()
    plt.xlabel('Prediction'); plt.ylabel('True Label')
    plt.title(f'Confusion Matrix — {tag}')
    plt.tight_layout()
    plt.savefig(f'cm_{tag}.png')
    plt.close()

    if was_batched:
        test_ds = test_ds.batch(batch_size)

    return acc, tpr, fpr, cm

def plot_history(history, tag):
    m = history.history
    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1)
    plt.semilogy(history.epoch, m['loss'],     label='train')
    plt.semilogy(history.epoch, m['val_loss'], label='val')
    plt.legend(); plt.ylabel('Loss'); plt.xlabel('Epoch')
    plt.subplot(2, 1, 2)
    plt.plot(history.epoch, m['accuracy'],     label='train')
    plt.plot(history.epoch, m['val_accuracy'], label='val')
    plt.legend(); plt.ylabel('Accuracy'); plt.xlabel('Epoch')
    plt.suptitle(tag); plt.tight_layout()
    plt.savefig(f'curves_{tag}.png')
    plt.close()
    print(f"Saved curves_{tag}.png")

# ── Model builders ────────────────────────────────────────────────────────────
def build_baseline(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_labels),
    ])
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

def build_overfit(input_shape):
    # wider and deeper version of the baseline to intentionally overfit for Q4
    # padding='same' keeps spatial dims from shrinking too fast on a 20x32 input
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(64,  3, activation='relu', padding='same'),
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, 3, activation='relu', padding='same'),
        layers.Conv2D(256, 3, activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(512, 3, activation='relu', padding='same'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(num_labels),
    ])
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

def build_with_reg(input_shape, reg_type='l2', coeff=0.01):
    reg = regularizers.l1(coeff) if reg_type == 'l1' else regularizers.l2(coeff)
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(64,  3, activation='relu', padding='same', kernel_regularizer=reg),
        layers.Conv2D(128, 3, activation='relu', padding='same', kernel_regularizer=reg),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, 3, activation='relu', padding='same', kernel_regularizer=reg),
        layers.Conv2D(256, 3, activation='relu', padding='same', kernel_regularizer=reg),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(512, 3, activation='relu', padding='same', kernel_regularizer=reg),
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu', kernel_regularizer=reg),
        layers.Dense(256, activation='relu', kernel_regularizer=reg),
        layers.Dense(num_labels),
    ])
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

def measure_sparsity(model, threshold_fraction=0.01):
    print("  Sparsity per layer:")
    for layer in model.layers:
        weights = layer.get_weights()
        if not weights:
            continue
        w = np.concatenate([wt.flatten() for wt in weights])
        threshold = threshold_fraction * np.max(np.abs(w))
        sparsity = np.mean(np.abs(w) < threshold)
        print(f"    {layer.name:35s}  sparsity={sparsity:.3f}  (n={len(w)})")

# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENTS
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== Building datasets ===")
train_ds_raw = preprocess_dataset(
    train_files,
    noisy_reps=[0.05, 0.1, 0.15, 0.2, 0.25, 0.1, 0.1, 0.1]
)
val_ds_raw  = preprocess_dataset(val_files)
test_ds_raw = preprocess_dataset(test_files)

train_ds = make_batched_ds(train_ds_raw, len(train_files))
val_ds   = make_batched_ds(val_ds_raw,   len(val_files))
test_ds  = make_batched_ds(test_ds_raw,  len(test_files))

for spec_batch, _ in train_ds.take(1):
    input_shape = spec_batch.shape[1:]
print('Input shape:', input_shape)

# ── Q2: Baseline ──────────────────────────────────────────────────────────────
print("\n=== Q2: Baseline model ===")
baseline = build_baseline(input_shape)
baseline.summary()
history_baseline = baseline.fit(train_ds, validation_data=val_ds, epochs=30)
plot_history(history_baseline, 'baseline')
acc_b, tpr_b, fpr_b, _ = evaluate_model(baseline, test_ds, tag='baseline')

# ── Q4: Overfit ───────────────────────────────────────────────────────────────
print("\n=== Q4: Overfit model ===")
overfit_model = build_overfit(input_shape)
history_overfit = overfit_model.fit(train_ds, validation_data=val_ds, epochs=30)
plot_history(history_overfit, 'overfit')
acc_o, tpr_o, fpr_o, _ = evaluate_model(overfit_model, test_ds, tag='overfit')

# ── Q5a: L2 ───────────────────────────────────────────────────────────────────
print("\n=== Q5a: L2 regularization ===")
l2_model = build_with_reg(input_shape, reg_type='l2', coeff=0.01)
history_l2 = l2_model.fit(train_ds, validation_data=val_ds, epochs=30)
plot_history(history_l2, 'l2_reg')
acc_l2, tpr_l2, fpr_l2, _ = evaluate_model(l2_model, test_ds, tag='l2_reg')

# ── Q5b: L1 ───────────────────────────────────────────────────────────────────
print("\n=== Q5b: L1 regularization ===")
l1_model = build_with_reg(input_shape, reg_type='l1', coeff=0.01)
history_l1 = l1_model.fit(train_ds, validation_data=val_ds, epochs=30)
plot_history(history_l1, 'l1_reg')
acc_l1, tpr_l1, fpr_l1, _ = evaluate_model(l1_model, test_ds, tag='l1_reg')

# ── Q6: Sparsity ──────────────────────────────────────────────────────────────
print("\n=== Q6: Sparsity ===")
print("--- Overfit model ---"); measure_sparsity(overfit_model)
print("--- L2 model ---");     measure_sparsity(l2_model)
print("--- L1 model ---");     measure_sparsity(l1_model)

# ── Q7: Limited data ──────────────────────────────────────────────────────────
print("\n=== Q7: Limited data (25 vs 250 samples) ===")
train_lim_raw = preprocess_dataset(
    train_files, noisy_reps=[0.1, 0.2],
    limit_cmd0=25, limit_cmd1=250
)
train_lim = make_batched_ds(train_lim_raw, 300)
lim_model = build_with_reg(input_shape, reg_type='l2', coeff=0.01)
history_lim = lim_model.fit(train_lim, validation_data=val_ds, epochs=30)
plot_history(history_lim, 'limited_data')
acc_lim, tpr_lim, fpr_lim, _ = evaluate_model(lim_model, test_ds, tag='limited_data')

# ── Q8: Data sweep ────────────────────────────────────────────────────────────
print("\n=== Q8: Data sweep for FPR<0.2, TPR>0.75 ===")
results_q8 = []
for n in [50, 100, 200, 400, 800]:
    print(f"\n  -- {n} samples per keyword --")
    ds_raw = preprocess_dataset(
        train_files, noisy_reps=[0.1, 0.2],
        limit_cmd0=n, limit_cmd1=n
    )
    ds_b = make_batched_ds(ds_raw, n * 2)
    m = build_with_reg(input_shape, reg_type='l2', coeff=0.01)
    m.fit(ds_b, validation_data=val_ds, epochs=25, verbose=0)
    _, tpr_q, fpr_q, _ = evaluate_model(m, test_ds, tag=f'q8_n{n}')
    results_q8.append({'n': n,
                       'tpr_w0': tpr_q[2], 'fpr_w0': fpr_q[2],
                       'tpr_w1': tpr_q[3], 'fpr_w1': fpr_q[3]})

print("\nQ8 Summary:")
print(f"{'N':>6}  {'TPR_'+commands[0]:>10}  {'FPR_'+commands[0]:>10}"
      f"  {'TPR_'+commands[1]:>10}  {'FPR_'+commands[1]:>10}")
for r in results_q8:
    print(f"{r['n']:>6}  {r['tpr_w0']:>10.3f}  {r['fpr_w0']:>10.3f}"
          f"  {r['tpr_w1']:>10.3f}  {r['fpr_w1']:>10.3f}")

print("\n=== All speech experiments complete ===")
