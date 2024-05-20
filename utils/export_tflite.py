import os

try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:
    import tensorflow as tf

    Interpreter, load_delegate = (
        tf.lite.Interpreter,
        tf.lite.experimental.load_delegate,
    )

from utils.opensource_dataloaders import normalize_img


def export_tflite(train_split, model, img_size: int, output_dir: str, save_name: str):
    # select representative dataset from training dataset
    representative_data = train_split.take(len(train_split))

    def representative_dataset_gen():
        for image, label in representative_data:
            image = normalize_img(image, label, img_size)[0]  # only take the image part
            image = tf.expand_dims(image, 0)
            yield [image]

    # convert to int8 TFLite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.representative_dataset = representative_dataset_gen
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model_int8 = converter.convert()

    # save int8 TFLite model
    print("Save INT8 TFLite model ...")
    int8_tflite_model_path = os.path.join(output_dir, f"{save_name}_int8.tflite")
    with open(int8_tflite_model_path, "wb") as f:
        f.write(tflite_model_int8)

    # convert to float32 TFLite format
    converter_float32 = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model_float32 = converter_float32.convert()

    # Save the model.
    print("Save FLOAT32 TFLite model ...")
    float32_tflite_model_path = os.path.join(output_dir, f"{save_name}_float32.tflite")
    with open(float32_tflite_model_path, "wb") as f:
        f.write(tflite_model_float32)

    # convert to float16 TFLite format
    converter_float16 = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_float16.target_spec.supported_types = [tf.float16]
    converter_float16.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model_float16 = converter_float16.convert()

    # Save the model.
    print("Save Float16 TFLite model ...")
    float16_tflite_model_path = os.path.join(output_dir, f"{save_name}_float16.tflite")
    with open(float16_tflite_model_path, "wb") as f:
        f.write(tflite_model_float16)

    return int8_tflite_model_path, float32_tflite_model_path, float16_tflite_model_path
