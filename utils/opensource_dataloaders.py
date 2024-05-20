import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import os

try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:
    import tensorflow as tf

    Interpreter, load_delegate = (
        tf.lite.Interpreter,
        tf.lite.experimental.load_delegate,
    )


def normalize_img(image, label, img_size: int):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (img_size, img_size))
    return image / 255.0, label


def opensource_dataloaders(
    dataset_name: str, img_size: int, shuffle_size: int, batch_size: int
):
    ds_data, ds_info = tfds.load(
        dataset_name,
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    num_of_class = ds_info.features["label"].num_classes

    train_split, test_split = ds_data["train"], ds_data["test"]

    print(f"number of train: {len(train_split)}")
    print(f"number of test: {len(test_split)}")

    ds_train = train_split.map(
        lambda image, label: normalize_img(image, label, img_size),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(shuffle_size)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = test_split.map(
        lambda image, label: normalize_img(image, label, img_size),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    ds_test = ds_test.batch(batch_size)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return num_of_class, train_split, ds_train, ds_test


def plot_history(history, output_dir: str, save_name: str):
    acc = history.history["sparse_categorical_accuracy"]
    val_acc = history.history["val_sparse_categorical_accuracy"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.axvline(10, ls="-.", color="magenta")
    plt.legend(loc="lower right")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")

    plt.subplot(2, 1, 2)
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.axvline(10, ls="-.", color="magenta")
    plt.legend(loc="upper right")
    plt.ylabel("Cross Entropy")
    plt.title("Training and Validation Loss")
    plt.xlabel("epoch")
    plt.show()
    plt.savefig(os.path.join(output_dir, f"{save_name}_training_history.png"))
