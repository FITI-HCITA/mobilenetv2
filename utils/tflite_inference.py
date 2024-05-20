import numpy as np
import time

try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:
    import tensorflow as tf

    Interpreter, load_delegate = (
        tf.lite.Interpreter,
        tf.lite.experimental.load_delegate,
    )


class TFLiteModel:
    def __init__(self, weight_file: str) -> None:
        self.interpreter = Interpreter(model_path=weight_file)  # load TFLite model
        self.interpreter.allocate_tensors()  # allocate
        self.input_details = self.interpreter.get_input_details()  # inputs
        self.output_details = self.interpreter.get_output_details()  # outputs

    def getInputSize(self):
        [n, inputH, inputW, c] = self.input_details[0]["shape"]
        return (inputH, inputW)

    def infer(self, im):
        input, output = self.input_details[0], self.output_details[0]
        scale, zero_point = input["quantization"]
        if input["dtype"] == np.int8:
            im = (im / scale + zero_point).astype(np.int8)  # de-scale
        elif input["dtype"] == np.uint8:
            im = (im / scale + zero_point).astype(np.uint8)  # de-scale
        elif input["dtype"] == np.float32:
            im = im.astype(np.float32)
        elif input["dtype"] == np.float16:
            im = im.astype(np.float16)
        # print("input data: ", im)
        self.interpreter.set_tensor(input["index"], im)
        self.interpreter.invoke()
        y = self.interpreter.get_tensor(output["index"])
        if input["dtype"] == np.int8 or input["dtype"] == np.uint8:
            scale, zero_point = output["quantization"]
            y = (y.astype(np.float32) - zero_point) * scale  # re-scale

        return y

    def accuracy(self, test_dataset):
        total, correct = 0, 0
        inference_time = []
        for images, lables in test_dataset:
            for i in range(len(images)):

                image = images[i].numpy()
                im = np.expand_dims(image, axis=0)  # Add batch dimension
                start_ms = time.time()

                predictions = self.infer(im)

                elapsed_ms = time.time() - start_ms
                inference_time.append(elapsed_ms * 1000.0)

                if lables[i].numpy() == np.argmax(predictions):
                    correct += 1
                total += 1

                if total % 50 == 0:
                    print(
                        "Accuracy after %i images: %f"
                        % (total, float(correct) / float(total))
                    )
        print(
            "Num image: {0:}, Accuracy: {1:.4f}, Latency: {2:.2f} ms".format(
                total, float(correct / total), np.array(inference_time).mean()
            )
        )
