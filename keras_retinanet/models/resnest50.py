from keras.layers import (
    Activation,
    Add,
    AveragePooling2D,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    Lambda,
    Layer,
    MaxPool2D,
    UpSampling2D,
    ZeroPadding2D,
    Reshape
)
from keras.models import Model
import tensorflow as tf
# from tensorflow.keras.activations import softmax
from tensorflow.keras.utils import get_custom_objects

class GroupedConv2D(Layer):
    COUNT=1
    def __init__(self, filters, kernel_size, **kwargs):
        """Initialize the layer.
        Args:
        filters: Integer, the dimensionality of the output space.
        kernel_size: An integer or a list. If it is a single integer, then it is
            same as the original Conv2D. If it is a list, then we split the channels
            and perform different kernel for each group.
        use_keras: An boolean value, whether to use keras layer.
        **kwargs: other parameters passed to the original conv2d layer.
        """
        super(GroupedConv2D, self).__init__(name=f'GroupConv2D_{GroupedConv2D.COUNT}')
        GroupedConv2D.COUNT += 1
        self.kernel_size = kernel_size
        self._groups = len(kernel_size)
        self._channel_axis = -1
        self.filters = filters
        splits = self._split_channels(filters, self._groups)
        self._kwargs = kwargs
        for i in range(self._groups):
            self.__setattr__(f'subconv{i}', self._get_conv2d(splits[i], kernel_size[i], **kwargs))

    def _get_conv2d(self, filters, kernel_size, **kwargs):
        """A helper function to create Conv2D layer."""
        return Conv2D(filters=filters, kernel_size=kernel_size, **kwargs)


    def _split_channels(self, total_filters, num_groups):
        split = [total_filters // num_groups for _ in range(num_groups)]
        split[0] += total_filters - sum(split)
        return split

    def call(self, inputs):
        if self._groups == 1:
            return self._convs[0](inputs)

        if tf.__version__ < "2.0.0":
            filters = inputs.shape[self._channel_axis].value
        else:
            filters = inputs.shape[self._channel_axis]
        splits = self._split_channels(filters, self._groups)
        x_splits = tf.split(inputs, splits, self._channel_axis)
        x_outputs = [self.__getattribute__(f'subconv{i}')(x) for i,x in enumerate(x_splits)]
        x = tf.concat(x_outputs, self._channel_axis)
        return x
    
    def compute_output_shape(self, input_shape):
        return (*input_shape[:3], self.filters)
    
    def get_config(self):
        config = super(GroupedConv2D, self).get_config()
        config.update({
          'filters': int(self.filters),
          'kernel_size':self.kernel_size,
#           "groups": self._groups,
#           'channel_axis': self._channel_axis,
#           'splits': self.splits,
          **self._kwargs,
        })
        return config

def _rsoftmax(input_tensor, filters, radix, groups):
    x = input_tensor
    batch = x.shape[0]
    if radix > 1:
        x = tf.reshape(x, [-1, groups, radix, filters // groups])
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.keras.activations.softmax(x, axis=1)
        x = tf.reshape(x, [-1, 1, 1, radix * filters])
    else:
        x = Activation("sigmoid")(x)
    return x

class _SplAtConv2d(Layer):
    COUNT=1
    def __init__(
        self,
        in_channels,
        filters=64,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        radix=0,
        active='relu',
        reduction_factor=4,
        **kwargs
        ):
        super(_SplAtConv2d, self).__init__(name=f'SplAtConv2D_{_SplAtConv2d.COUNT}')
        _SplAtConv2d.COUNT += 1
        self.in_channels = in_channels
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.radix = radix
        self.active = active
        self.reduction_factor = reduction_factor
        self.inter_channels = max(int(self.in_channels) * self.radix
                                  // self.reduction_factor, 32)
        self.channel_axis = -1  # not for change
        self.group_conv = GroupedConv2D(
            filters=self.filters * self.radix,
            kernel_size=[self.kernel_size for i in range(self.groups * self.radix)],
            padding='same',
            kernel_initializer='he_normal',
            use_bias=False,
            data_format='channels_last',
            dilation_rate=self.dilation,
            )
        self.BN0 = BatchNormalization(axis=self.channel_axis,
                                      epsilon=1.001e-5)
        self.BN1 = BatchNormalization(axis=self.channel_axis,
                                      epsilon=1.001e-5)
        self.CONV0 = Conv2D(self.inter_channels, kernel_size=1)
        self.CONV1 = Conv2D(filters * radix, kernel_size=1)
        self.ACT = Activation(self.active)
        self.GAP = GlobalAveragePooling2D(data_format='channels_last')
        self.RESHAPE = Reshape([1, 1, filters])
        self._kwargs = kwargs
    
    def call(self, inputs, **kwargs):
        x = inputs
        in_channels = int(x.shape[-1])

        x = self.group_conv(x)

        x = self.BN0(x)
        x = self.ACT(x)

        self.output_shapes = x.get_shape().as_list()[1:3]
        batch, rchannel = x.shape[0], x.shape[-1]
        if self.radix > 1:
            splited = tf.split(x, self.radix, axis=-1)
            gap = sum(splited)
        else:
            gap = x

        #print('sum',gap.shape)
        gap = self.GAP(gap)
        #print('after gap ', gap.shape)
        gap = self.RESHAPE(gap)
        #print('adaptive_avg_pool2d',gap.shape)

        x = self.CONV0(gap)

        x = self.BN1(x)
        x = self.ACT(x)
        x = self.CONV1(x)

        atten = _rsoftmax(x, self.filters, self.radix, self.groups)

        if self.radix > 1:
            logits = tf.split(atten, self.radix, axis=-1)
            out = sum([a * b for a, b in zip(splited, logits)])
        else:
            out = atten * x
        return out
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], *self.output_shapes, self.filters)
    
    def get_config(self):
        config = super(_SplAtConv2d, self).get_config()
        config.update({
          "in_channels":int(self.in_channels),
          "filters":int(self.filters),
          "kernel_size":int(self.kernel_size),
          "stride":int(self.stride),
          "dilation":int(self.dilation),     
          "radix": int(self.radix),
          'groups': int(self.groups),
          'active': self.active,
          'reduction_factor':int(self.reduction_factor),
          **self._kwargs
#           'inter_channels': self.inter_channels,
#           'channel_axis': self.channel_axis,          
#           'group_conv': self.group_conv,
#           'BN0': self.BN0,
#           'BN1': self.BN1,
#           'CONV0': self.CONV0,
#           'CONV1': self.CONV1,
#           'ACT': self.ACT,
#           'GAP': self.GAP,
#           'RESHAPE': self.RESHAPE,
        })
        return config

def get_flops(model):
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.compat.v1.profiler.profile(
        graph=tf.compat.v1.keras.backend.get_session().graph, run_meta=run_meta, cmd="op", options=opts
    )

    return flops.total_float_ops  # Prints the "flops" of the model.


class Mish(Activation):
    """
    based on https://github.com/digantamisra98/Mish/blob/master/Mish/TFKeras/mish.py
    Mish Activation Function.
    """

    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = "Mish"


def mish(inputs):
    # with tf.device("CPU:0"):
    result = inputs * tf.math.tanh(tf.math.softplus(inputs))
    return result


class ResNest():
    def __init__(self, verbose=False, active="relu", n_classes=81,
                 dropout_rate=0.2, fc_activation=None, blocks_set=[3, 4, 6, 3], radix=2, groups=1,
                 bottleneck_width=64, deep_stem=True, stem_width=32, block_expansion=4, avg_down=True,
                 avd=True, avd_first=False, preact=False, using_basic_block=False,using_cb=False):
        self.channel_axis = -1  # not for change
        self.verbose = verbose
        self.active = active  # default relu
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate
        self.fc_activation = fc_activation

        self.blocks_set = blocks_set
        self.radix = radix
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width

        self.deep_stem = deep_stem
        self.stem_width = stem_width
        self.block_expansion = block_expansion
        self.avg_down = avg_down
        self.avd = avd
        self.avd_first = avd_first

        # self.cardinality = 1
        self.dilation = 1
        self.preact = preact
        self.using_basic_block = using_basic_block
        self.using_cb = using_cb

    def _make_stem(self, input_tensor, stem_width=64, deep_stem=False):
        x = input_tensor
        if deep_stem:
            x = Conv2D(stem_width, kernel_size=3, strides=2, padding="same", kernel_initializer="he_normal",
                       use_bias=False, data_format="channels_last")(x)

            x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
            x = Activation(self.active)(x)

            x = Conv2D(stem_width, kernel_size=3, strides=1, padding="same",
                       kernel_initializer="he_normal", use_bias=False, data_format="channels_last")(x)

            x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
            x = Activation(self.active)(x)

            x = Conv2D(stem_width * 2, kernel_size=3, strides=1, padding="same", kernel_initializer="he_normal",
                       use_bias=False, data_format="channels_last")(x)

            # x = BatchNormalization(axis=self.channel_axis,epsilon=1.001e-5)(x)
            # x = Activation(self.active)(x)
        else:
            x = Conv2D(stem_width, kernel_size=7, strides=2, padding="same", kernel_initializer="he_normal",
                       use_bias=False, data_format="channels_last")(x)
            # x = BatchNormalization(axis=self.channel_axis,epsilon=1.001e-5)(x)
            # x = Activation(self.active)(x)
        return x

    def _make_block(
        self, input_tensor, first_block=True, filters=64, stride=2, radix=1, avd=False, avd_first=False, is_first=False
    ):
        x = input_tensor
        inplanes = input_tensor.shape[-1]
        if stride != 1 or inplanes != filters * self.block_expansion:
            short_cut = input_tensor
            if self.avg_down:
                if self.dilation == 1:
                    short_cut = AveragePooling2D(pool_size=stride, strides=stride, padding="same", data_format="channels_last")(
                        short_cut
                    )
                else:
                    short_cut = AveragePooling2D(pool_size=1, strides=1, padding="same", data_format="channels_last")(short_cut)
                short_cut = Conv2D(filters * self.block_expansion, kernel_size=1, strides=1, padding="same",
                                   kernel_initializer="he_normal", use_bias=False, data_format="channels_last")(short_cut)
            else:
                short_cut = Conv2D(filters * self.block_expansion, kernel_size=1, strides=stride, padding="same",
                                   kernel_initializer="he_normal", use_bias=False, data_format="channels_last")(short_cut)

            short_cut = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(short_cut)
        else:
            short_cut = input_tensor

        group_width = int(filters * (self.bottleneck_width / 64.0)) * self.cardinality
        x = Conv2D(group_width, kernel_size=1, strides=1, padding="same", kernel_initializer="he_normal", use_bias=False,
                   data_format="channels_last")(x)
        x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
        x = Activation(self.active)(x)

        avd = avd and (stride > 1 or is_first)
        avd_first = avd_first

        if avd:
            avd_layer = AveragePooling2D(pool_size=3, strides=stride, padding="same", data_format="channels_last")
            stride = 1

        if avd and avd_first:
            x = avd_layer(x)

        if radix >= 1:
            x = _SplAtConv2d(x.shape[-1], filters=group_width, kernel_size=3, stride=stride, dilation=self.dilation,
                                  groups=self.cardinality, radix=radix, active=self.active)(x)
        else:
            x = Conv2D(group_width, kernel_size=3, strides=stride, padding="same", kernel_initializer="he_normal",
                       dilation_rate=self.dilation, use_bias=False, data_format="channels_last")(x)
            x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
            x = Activation(self.active)(x)

        if avd and not avd_first:
            x = avd_layer(x)
            # print('can')
        x = Conv2D(filters * self.block_expansion, kernel_size=1, strides=1, padding="same", kernel_initializer="he_normal",
                   dilation_rate=self.dilation, use_bias=False, data_format="channels_last")(x)
        x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)

        m2 = Add()([x, short_cut])
        m2 = Activation(self.active)(m2)
        return m2

    def _make_block_basic(
        self, input_tensor, first_block=True, filters=64, stride=2, radix=1, avd=False, avd_first=False, is_first=False
    ):
        """Conv2d_BN_Relu->Bn_Relu_Conv2d
        """
        x = input_tensor
        x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
        x = Activation(self.active)(x)

        short_cut = x
        inplanes = input_tensor.shape[-1]
        if stride != 1 or inplanes != filters * self.block_expansion:
            if self.avg_down:
                if self.dilation == 1:
                    short_cut = AveragePooling2D(pool_size=stride, strides=stride, padding="same", data_format="channels_last")(
                        short_cut
                    )
                else:
                    short_cut = AveragePooling2D(pool_size=1, strides=1, padding="same", data_format="channels_last")(short_cut)
                short_cut = Conv2D(filters, kernel_size=1, strides=1, padding="same", kernel_initializer="he_normal",
                                   use_bias=False, data_format="channels_last")(short_cut)
            else:
                short_cut = Conv2D(filters, kernel_size=1, strides=stride, padding="same", kernel_initializer="he_normal",
                                   use_bias=False, data_format="channels_last")(short_cut)

        group_width = int(filters * (self.bottleneck_width / 64.0)) * self.cardinality
        avd = avd and (stride > 1 or is_first)
        avd_first = avd_first

        if avd:
            avd_layer = AveragePooling2D(pool_size=3, strides=stride, padding="same", data_format="channels_last")
            stride = 1

        if avd and avd_first:
            x = avd_layer(x)

        if radix >= 1:
            x = _SplAtConv2d(x.shape[-1], filters=group_width, kernel_size=3, stride=stride, dilation=self.dilation,
                                  groups=self.cardinality, radix=radix, active=self.active)(x)
        else:
            x = Conv2D(filters, kernel_size=3, strides=stride, padding="same", kernel_initializer="he_normal",
                       dilation_rate=self.dilation, use_bias=False, data_format="channels_last")(x)

        if avd and not avd_first:
            x = avd_layer(x)
            # print('can')

        x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
        x = Activation(self.active)(x)
        x = Conv2D(filters, kernel_size=3, strides=1, padding="same", kernel_initializer="he_normal",
                   dilation_rate=self.dilation, use_bias=False, data_format="channels_last")(x)
        m2 = Add()([x, short_cut])
        return m2

    def _make_layer(self, input_tensor, blocks=4, filters=64, stride=2, is_first=True):
        x = input_tensor
        if self.using_basic_block is True:
            x = self._make_block_basic(x, first_block=True, filters=filters, stride=stride, radix=self.radix,
                                       avd=self.avd, avd_first=self.avd_first, is_first=is_first)
            # print('0',x.shape)

            for i in range(1, blocks):
                x = self._make_block_basic(
                    x, first_block=False, filters=filters, stride=1, radix=self.radix, avd=self.avd, avd_first=self.avd_first
                )
                # print(i,x.shape)

        elif self.using_basic_block is False:
            x = self._make_block(x, first_block=True, filters=filters, stride=stride, radix=self.radix, avd=self.avd,
                                 avd_first=self.avd_first, is_first=is_first)
            # print('0',x.shape)

            for i in range(1, blocks):
                x = self._make_block(
                    x, first_block=False, filters=filters, stride=1, radix=self.radix, avd=self.avd, avd_first=self.avd_first
                )
                # print(i,x.shape)
        return x

    def _make_Composite_layer(self,input_tensor,filters=256,kernel_size=1,stride=1,upsample=True):
        x = input_tensor
        x = Conv2D(filters, kernel_size, strides=stride, use_bias=False)(x)
        x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
        if upsample:
            x = UpSampling2D(size=2)(x)
        return x

    def build(self, inputs):
        output_tensors = []
        get_custom_objects().update({'mish': Mish(mish)})

        input_sig = inputs
        x = self._make_stem(input_sig, stem_width=self.stem_width, deep_stem=self.deep_stem)

        if self.preact is False:
            x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
            x = Activation(self.active)(x)
        if self.verbose:
            print("stem_out", x.shape)

        x = MaxPool2D(pool_size=3, strides=2, padding="same", data_format="channels_last")(x)
        if self.verbose:
            print("MaxPool2D out", x.shape)

        if self.preact is True:
            x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
            x = Activation(self.active)(x)
        
        x = self._make_layer(x, blocks=self.blocks_set[0], filters=64, stride=1, is_first=False)
        if self.verbose:
            print("-" * 5, "layer 0 out", x.shape, "-" * 5)

        b1_b3_filters = [64,128,256,512]
        for i in range(3):
            idx = i+1
            x = self._make_layer(x, blocks=self.blocks_set[idx], filters=b1_b3_filters[idx], stride=2)
            if self.verbose: print('----- layer {} out {} -----'.format(idx,x.shape))
            output_tensors.append(x)  # outputs for retinanet

#         x = GlobalAveragePooling2D(name='avg_pool')(x) 
#         if self.verbose:
#             print("pool_out:", x.shape) # remove the concats var

#         if self.dropout_rate > 0:
#             x = Dropout(self.dropout_rate, noise_shape=None)(x)

#         fc_out = Dense(self.n_classes, kernel_initializer="he_normal", use_bias=False, name="fc_NObias")(x) # replace concats to x
#         if self.verbose:
#             print("fc_out:", fc_out.shape)

#         if self.fc_activation:
#             fc_out = Activation(self.fc_activation)(fc_out)

        model = Model(inputs=input_sig, outputs=output_tensors)

        if self.verbose:
            print("Resnest builded with input {}, output{}".format(input_sig.shape, fc_out.shape))
        if self.verbose:
            print("-------------------------------------------")
        if self.verbose:
            print("")
        if self.verbose:
            print(model.summary())

        return model
