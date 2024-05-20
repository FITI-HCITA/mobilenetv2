try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:
    import tensorflow as tf
    Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate,


def mobilenetV2(shape, ratio = 1):
    input_node = tf.keras.layers.Input(shape=shape)

    net = tf.keras.layers.Conv2D(int(32*ratio), 3, (2, 2), use_bias=False, padding='same')(input_node)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.ReLU(max_value=6)(net)

    net = tf.keras.layers.DepthwiseConv2D(3, use_bias=False, padding='same')(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.ReLU(max_value=6)(net)
    net = tf.keras.layers.Conv2D(int(16*ratio), 1, use_bias=False, padding='same')(net)
    net = tf.keras.layers.BatchNormalization()(net)

    net = bottleneck(net, int(16*ratio), int(24*ratio), (2, 2), shortcut=False, zero_pad=True)  # block_1
    net = bottleneck(net, int(24*ratio), int(24*ratio), (1, 1), shortcut=True)  # block_2

    net = bottleneck(net, int(24*ratio), int(32*ratio), (2, 2), shortcut=False, zero_pad=True)  # block_3
    net = bottleneck(net, int(32*ratio), int(32*ratio), (1, 1), shortcut=True)  # block_4
    net = bottleneck(net, int(32*ratio), int(32*ratio), (1, 1), shortcut=True)  # block_5

    net = bottleneck(net, int(32*ratio), int(64*ratio), (2, 2), shortcut=False, zero_pad=True)  # block_6
    net = bottleneck(net, int(64*ratio), int(64*ratio), (1, 1), shortcut=True)  # block_7
    net = bottleneck(net, int(64*ratio), int(64*ratio), (1, 1), shortcut=True)  # block_8
    net = bottleneck(net, int(64*ratio), int(64*ratio), (1, 1), shortcut=True)  # block_9

    net = bottleneck(net, int(64*ratio), int(96*ratio), (1, 1), shortcut=False)  # block_10
    net = bottleneck(net, int(96*ratio), int(96*ratio), (1, 1), shortcut=True)  # block_11
    net = bottleneck(net, int(96*ratio), int(96*ratio), (1, 1), shortcut=True)  # block_12

    net = bottleneck(net, int(96*ratio), int(160*ratio), (2, 2), shortcut=False, zero_pad=True)  # block_13
    net = bottleneck(net, int(160*ratio), int(160*ratio), (1, 1), shortcut=True)  # block_14
    net = bottleneck(net, int(160*ratio), int(160*ratio), (1, 1), shortcut=True)  # block_15

    net = bottleneck(net, int(160*ratio), int(320*ratio), (1, 1), shortcut=False)  # block_16

    net = tf.keras.layers.Conv2D(int(1280*ratio), 1, use_bias=False, padding='same')(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.ReLU(max_value=6)(net)

    return input_node, net

def bottleneck(net, filters, out_ch, strides, shortcut=True, zero_pad=False):

    padding = 'valid' if zero_pad else 'same'
    shortcut_net = net

    net = tf.keras.layers.Conv2D(filters * 6, 1, use_bias=False, padding='same')(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.ReLU(max_value=6)(net)
    if zero_pad:
        net = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1)))(net)

    net = tf.keras.layers.DepthwiseConv2D(3, strides=strides, use_bias=False, padding=padding)(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.ReLU(max_value=6)(net)

    net = tf.keras.layers.Conv2D(out_ch, 1, use_bias=False, padding='same')(net)
    net = tf.keras.layers.BatchNormalization()(net)

    if shortcut:
        net = tf.keras.layers.Add()([net, shortcut_net])

    return net