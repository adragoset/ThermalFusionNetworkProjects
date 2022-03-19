"""MobileNet v3 models for Keras.
The following table describes the performance of MobileNets:
------------------------------------------------------------------------
MACs stands for Multiply Adds
| Classification Checkpoint| MACs(M)| Parameters(M)| Top1 Accuracy| Pixel1 CPU(ms)|
| [mobilenet_v3_large_1.0_224]              | 217 | 5.4 |   75.6   |   51.2   |
| [mobilenet_v3_large_0.75_224]             | 155 | 4.0 |   73.3   |   39.8   |
| [mobilenet_v3_large_minimalistic_1.0_224] | 209 | 3.9 |   72.3   |   44.1   |
| [mobilenet_v3_small_1.0_224]              | 66  | 2.9 |   68.1   |   15.8   |
| [mobilenet_v3_small_0.75_224]             | 44  | 2.4 |   65.4   |   12.8   |
| [mobilenet_v3_small_minimalistic_1.0_224] | 65  | 2.0 |   61.9   |   12.2   |
The weights for all 6 models are obtained and
translated from the Tensorflow checkpoints
from TensorFlow checkpoints found [here]
(https://github.com/tensorflow/models/tree/master/research/
slim/nets/mobilenet/README.md).
# Reference
This file contains building code for MobileNetV3, based on
[Searching for MobileNetV3]
(https://arxiv.org/pdf/1905.02244.pdf) (ICCV 2019)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import MobileNetMultiSpectralFusionHelpers
from tensorflow.keras import activations


backend = None
layers = None
models = None
keras_utils = None

BASE_WEIGHT_PATH = ('https://github.com/DrSlink/mobilenet_v3_keras/'
                    'releases/download/v1.0/')
WEIGHTS_HASHES = {
    'large_224_0.75_float': (
        '765b44a33ad4005b3ac83185abf1d0eb',
        'c256439950195a46c97ede7c294261c6'),
    'large_224_1.0_float': (
        '59e551e166be033d707958cf9e29a6a7',
        '12c0a8442d84beebe8552addf0dcb950'),
    'large_minimalistic_224_1.0_float': (
        '675e7b876c45c57e9e63e6d90a36599c',
        'c1cddbcde6e26b60bdce8e6e2c7cae54'),
    'small_224_0.75_float': (
        'cb65d4e5be93758266aa0a7f2c6708b7',
        'c944bb457ad52d1594392200b48b4ddb'),
    'small_224_1.0_float': (
        '8768d4c2e7dee89b9d02b2d03d65d862',
        '5bec671f47565ab30e540c257bba8591'),
    'small_minimalistic_224_1.0_float': (
        '99cd97fb2fcdad2bf028eb838de69e37',
        '1efbf7e822e03f250f45faa3c6bbe156'),
}


# This function is taken from the original tf repo.
# It ensures that all layers have a channel number that is divisible by 8
# It can be seen here:
# https://github.com/tensorflow/models/blob/master/research/
# slim/nets/mobilenet/mobilenet.py


def MobileMultiSpectralFusionNet(stack_fn,
                last_point_ch,
                input_shape=None,
                alpha=1.0,
                weights=None,
                input_tensor=None,
                pooling=None,
                **kwargs):
    """Instantiates the MobileNetV3 architecture.
    # Arguments
        stack_fn: a function that returns output tensor for the
            stacked residual blocks.
        last_point_ch: number channels at the last layer (before top)
        input_shape: optional shape tuple, to be specified if you would
            like to use a model with an input img resolution that is not
            (224, 224, 3).
            It should have exactly 3 inputs channels (224, 224, 3).
            You can also omit this option if you would like
            to infer input_shape from an input_tensor.
            If you choose to include both input_tensor and input_shape then
            input_shape will be used if they match, if the shapes
            do not match then we will throw an error.
            E.g. `(160, 160, 3)` would be one valid value.
        alpha: controls the width of the network. This is known as the
            depth multiplier in the MobileNetV3 paper, but the name is kept for
            consistency with MobileNetV1 in Keras.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                are used at each layer.
        model_type: MobileNetV3 is defined as two models: large and small. These
        models are targeted at high and low resource use cases respectively.
        minimalistic: In addition to large and small models this module also contains
            so-called minimalistic models, these models have the same per-layer
            dimensions characteristic as MobilenetV3 however, they don't utilize any
            of the advanced blocks (squeeze-and-excite units, hard-swish, and 5x5
            convolutions). While these models are less efficient on CPU, they are
            much more performant on GPU/DSP.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        dropout_rate: fraction of the input units to drop on the last layer
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid model type, argument for `weights`,
            or invalid input shape when weights='imagenet'
    """
    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = MobileNetMultiSpectralFusionHelpers.get_submodules_from_kwargs(kwargs)

    if not (weights in {None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) '
                         'or the path to the weights file to be loaded.')

    # Determine proper input shape and default size.
    # If both input_shape and input_tensor are used, they should match
    if input_shape is not None and input_tensor is not None:
        try:
            is_input_t_tensor = backend.is_keras_tensor(input_tensor)
        except ValueError:
            try:
                is_input_t_tensor = backend.is_keras_tensor(
                    keras_utils.get_source_inputs(input_tensor))
            except ValueError:
                raise ValueError('input_tensor: ', input_tensor,
                                 'is not type input_tensor')
        if is_input_t_tensor:
            if backend.image_data_format == 'channels_first':
                if backend.int_shape(input_tensor)[1] != input_shape[1]:
                    raise ValueError('input_shape: ', input_shape,
                                     'and input_tensor: ', input_tensor,
                                     'do not meet the same shape requirements')
            else:
                if backend.int_shape(input_tensor)[2] != input_shape[1]:
                    raise ValueError('input_shape: ', input_shape,
                                     'and input_tensor: ', input_tensor,
                                     'do not meet the same shape requirements')
        else:
            raise ValueError('input_tensor specified: ', input_tensor,
                             'is not a keras tensor')

    # If input_shape is None, infer shape from input_tensor
    if input_shape is None and input_tensor is not None:

        try:
            backend.is_keras_tensor(input_tensor)
        except ValueError:
            raise ValueError('input_tensor: ', input_tensor,
                             'is type: ', type(input_tensor),
                             'which is not a valid type')

        if backend.is_keras_tensor(input_tensor):
            if backend.image_data_format() == 'channels_first':
                rows = backend.int_shape(input_tensor)[2]
                cols = backend.int_shape(input_tensor)[3]
                channels = backend.int_shape(input_tensor)[1]
                input_shape = (channels, cols, rows)
            else:
                rows = backend.int_shape(input_tensor)[1]
                cols = backend.int_shape(input_tensor)[2]
                channels = backend.int_shape(input_tensor)[3]
                input_shape = (cols, rows, channels)
    # If input_shape is None and input_tensor is None using standart shape
    if input_shape is None and input_tensor is None:
        input_shape = (None, None, 4)

    if backend.image_data_format() == 'channels_last':
        row_axis, col_axis = (0, 1)
    else:
        row_axis, col_axis = (1, 2)
    rows = input_shape[row_axis]
    cols = input_shape[col_axis]
    if rows and cols and (rows < 32 or cols < 32):
        raise ValueError('Input size must be at least 32x32; got `input_shape=' +
                         str(input_shape) + '`')

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

    kernel = 3
    activation = activations.sigmoid
    se_ratio = None

    x = layers.ZeroPadding2D(padding=MobileNetMultiSpectralFusionHelpers.correct_pad(backend, img_input, 3),
                             name='Conv_pad')(img_input)
    x = layers.Conv2D(16,
                      kernel_size=3,
                      strides=(2, 2),
                      padding='valid',
                      use_bias=False,
                      name='Conv')(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  epsilon=1e-3,
                                  momentum=0.999,
                                  name='Conv/BatchNorm')(x)
    x = layers.Activation(activation)(x)

    x = stack_fn(x, kernel, activation, se_ratio)

    last_conv_ch = MobileNetMultiSpectralFusionHelpers.depth(backend.int_shape(x)[channel_axis] * 6)

    # if the width multiplier is greater than 1 we
    # increase the number of output channels
    if alpha > 1.0:
        last_point_ch = MobileNetMultiSpectralFusionHelpers.depth(last_point_ch * alpha)

    x = layers.Conv2D(last_conv_ch,
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      name='Conv_1')(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  epsilon=1e-3,
                                  momentum=0.999,
                                  name='Conv_1/BatchNorm')(x)
    x = layers.Activation(activation)(x)

    if pooling == 'avg':
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    elif pooling == 'max':
        x = layers.GlobalMaxPooling2D(name='max_pool')(x)
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = models.Model(inputs, x, name='MobileMultiSpectralFusionNetDefault')

    if weights is not None:
        model.load_weights(weights)

    return model


def MobileMultiSpectralFusionNetDefault(input_shape=None,
                     alpha=1.0,
                     weights=None,
                     input_tensor=None,
                     pooling=None,
                     dropout_rate=0.2,
                     **kwargs):
    def stack_fn(x, kernel, activation, se_ratio):
        def depth(d):
            return depth(d * alpha) 
        x = MobileNetMultiSpectralFusionHelpers.inverted_res_block(layers, backend, x, 1, depth(32), kernel, 1, se_ratio, activation, 0)
        return x
    return MobileMultiSpectralFusionNet(stack_fn,
                       1024,
                       input_shape,
                       alpha,
                       weights,
                       input_tensor,
                       pooling,
                       **kwargs)


setattr(MobileMultiSpectralFusionNetDefault, '__doc__', MobileMultiSpectralFusionNetDefault.__doc__)