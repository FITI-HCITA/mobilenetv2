try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:
    import tensorflow as tf

    Interpreter, load_delegate = (
        tf.lite.Interpreter,
        tf.lite.experimental.load_delegate,
    )

import os
import datetime
import argparse

from models.mobilenetV2 import mobilenetV2
from utils.opensource_dataloaders import opensource_dataloaders, plot_history
from utils.export_tflite import export_tflite
from utils.tflite_inference import TFLiteModel


def train(
    image_size: int,
    num_of_class: int,
    ratio: int,
    LR: float,
    epochs: int,
    ds_train,
    ds_test,
):
    # define model
    input_node, net = mobilenetV2((image_size, image_size, 3), ratio=ratio)
    net = tf.keras.layers.GlobalAveragePooling2D()(net)
    net = tf.keras.layers.Dense(num_of_class, activation="softmax")(net)

    model = tf.keras.Model(inputs=[input_node], outputs=[net])
    model.summary()

    # compile model
    model.compile(
        optimizer=tf.keras.optimizers.SGD(LR),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    # train model
    history = model.fit(ds_train, epochs=epochs, validation_data=ds_test, verbose=True)

    return model, history


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="horses_or_humans",
        help="opensource in tensorflow_datasets",
    )
    parser.add_argument("--img_size", type=int, default=224, help="input image size")
    parser.add_argument(
        "--epochs", type=int, default=500, help="number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="number of training batch size"
    )
    parser.add_argument(
        "--shuffle_size", type=int, default=1000, help="number of shuffle size"
    )
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.5,
        help="channel ratio of MobilenetV2. original ratio = 1",
    )
    args = parser.parse_args()

    IMG_SIZE = args.img_size
    SHUFFLE_SIZE = args.shuffle_size
    BATCH_SIZE = args.batch_size
    LR = args.lr
    EPOCHS = args.epochs
    SAVE_NAME = "mobilenetv2_w{}{}_{}".format(
        str(args.ratio).split(".")[0], str(args.ratio).split(".")[-1], args.dataset
    )
    OUTPUT_DIR = os.path.join("output", f"{SAVE_NAME}_{str(datetime.date.today())}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # prepare datasets
    num_of_class, train_split, ds_train, ds_test = opensource_dataloaders(
        dataset_name=args.dataset,
        img_size=IMG_SIZE,
        shuffle_size=SHUFFLE_SIZE,
        batch_size=BATCH_SIZE,
    )

    # training
    model, history = train(
        image_size=IMG_SIZE,
        num_of_class=num_of_class,
        ratio=args.ratio,
        LR=LR,
        epochs=EPOCHS,
        ds_train=ds_train,
        ds_test=ds_test,
    )

    # plot the training history
    plot_history(history, output_dir=OUTPUT_DIR, save_name=SAVE_NAME)

    # Evaluate trained model
    _, trained_model_accuracy = model.evaluate(ds_test, verbose=True)
    print("trained model test accuracy:", trained_model_accuracy)

    # save h5 model
    print("Save H5 model ...")
    model.save(os.path.join(OUTPUT_DIR, f"{SAVE_NAME}_weights.h5"), save_format="h5")

    # Export TFLite model
    int8_tflite_model_path, float32_tflite_model_path, float16_tflite_model_path = (
        export_tflite(
            train_split,
            model,
            img_size=IMG_SIZE,
            output_dir=OUTPUT_DIR,
            save_name=SAVE_NAME,
        )
    )

    # inference TFLite model
    print("Inference INT8 TFLite model ...")
    tflite_model = TFLiteModel(int8_tflite_model_path)
    tflite_model.accuracy(ds_test)

    print("Inference FLOAT16 TFLite model ...")
    tflite_model = TFLiteModel(float16_tflite_model_path)
    tflite_model.accuracy(ds_test)

    print("Inference FLOAT32 TFLite model ...")
    tflite_model = TFLiteModel(float32_tflite_model_path)
    tflite_model.accuracy(ds_test)
