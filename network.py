from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, BatchNormalization, LeakyReLU, Concatenate
from tensorflow.keras.models import Model
import tensorflow as tf
def conv_batch_maxpooling(input_tensor, output_channels, kernel_size, 
                          pooling_size, func=LeakyReLU, batch_normalization=True,
                          use_bias=False
                          ):
    """
    This function creates a convolutional unit made up of a convolution followed by
    a second convolution and a pooling layer. The depth (number of channels) in each
    convolution is the same, as well as the filter size. Activation function may
    be provided. Default is the 'LeakyReLU'. Default settings use batch normalization
    which could be turned off. padding will be set to 'same' due to compatibility
    since the original code of @sponka at 
    https://github.com/Sponka/Astro_U-net/blob/master/code/create_new_image.py#L20
    uses tensorflow.contrib.slim.conv2d which defaults to 'same'

    args:
        input_tensor : input object
        output_channels : number of output channels for the layer
        kernel_size : shape of the filter
        pooling_size : pooling size
        func : activation function default is Leaky ReLU
        batch_normalization : boolean as to whether to apply batch normalization. 
                                Default is True
    returns:
        conv : convolutional layer after twice the operation
        pool : pooling layer at the end end of the convolution
    """
    if batch_normalization:
        conv = Conv2D(filters=output_channels, kernel_size=kernel_size, activation=None, padding='same')(input_tensor)
        bn = BatchNormalization()(conv)
        conv = func()(bn)
        conv = Conv2D(filters=output_channels, kernel_size=kernel_size, activation=None, padding='same')(conv)
        bn = BatchNormalization()(conv)
        conv = func()(bn)
        pool = MaxPooling2D(pooling_size, padding='same')(conv)
    else:
        conv = Conv2D(filters=output_channels, kernel_size=kernel_size, activation=func, padding='same', use_bias=use_bias)(input_tensor)
        conv = Conv2D(filters=output_channels, kernel_size=kernel_size, activation=func, padding='same', use_bias=use_bias)(conv)
        pool = MaxPooling2D(pooling_size, padding='same')(conv)
    return conv, pool

def upsample_and_concat(input_layer, previous_layer, kernel_size, filter_size, exp_time=None, exp=False,
                        func=LeakyReLU, batch_normalization=True, use_bias=False):
    output_channels = previous_layer.shape[-1] #Reading off the number of channels in the corresponding
                                               #Layer so as to be able to concatenate both together

    """
    So the idea is to increase the size (H, W) of an image while decreasing
    eventually the number of filters/channels (C). Similar to 'conv_batch_maxpooling'
    the convolutional layers uses 'same' padding due to the compatibility issues with
    the original code

    args:
        input_layer : the input layer
        previous_layer : the layer with which to concatenate the deconvolution
        filter_size : the size of the window to be used while pooling
        kerner_size : the size of the filter windows for the normal convolutions
        exp_time : I am not sure of this yet
    returns:
        deconv : deconvolved and concatenated layer

    """
    # N, H, W, C Order is used
    #'kernel_size' is the parameter for the size of the sliding window/filter/kernel
    #'strides' determines the factor of upsampling
    deconv = Conv2DTranspose(
        filters=output_channels,
        kernel_size=filter_size,
        strides=filter_size,
        padding='same',
        kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0, stddev=0.02)
    )(input_layer)
    
    print(output_channels)
    print(deconv.shape)
    print(previous_layer.shape)

    # The previous layer and the convolved layer are concatenated along the channels (last index)
    # meaning the number of channels will increase
    if exp:
        # Generate a constant tensor and add it as an additional channel
        cons = tf.fill(tf.shape(deconv), exp_time)
        c = tf.slice(cons, [0, 0, 0, 0], [-1, -1, -1, 1])  # Extract one channel
        deconv = Concatenate(axis=-1)([deconv, previous_layer, c])
        deconv.set_shape([None, None, None, output_channels * 2 + 1])
    else:
        # Standard concatenation
        deconv = Concatenate(axis=-1)([deconv, previous_layer])

    if batch_normalization:
        deconv = Conv2D(filters=output_channels, kernel_size=kernel_size, activation=None, padding='same')(deconv)
        deconv = BatchNormalization()(deconv)
        deconv = func()(deconv)
        deconv = Conv2D(filters=output_channels, kernel_size=kernel_size, activation=None, padding='same')(deconv)
        deconv = BatchNormalization()(deconv)
        deconv = func()(deconv)
    else:
        deconv = Conv2D(filters=output_channels, kernel_size=kernel_size, activation=func, padding='same', use_bias=use_bias)(deconv)
        deconv = Conv2D(filters=output_channels, kernel_size=kernel_size, activation=func, padding='same', use_bias=use_bias)(deconv)
    return deconv

def network(input_shape, depth, kernel_size, filter_size, pooling_size, n_of_initial_channels,
            func=LeakyReLU, batch_normalization=True, use_bias=False, exp_time=None, exp=False):
    """
    
    args:
        input_shape : input shape without N (number of samples)
        depth : the depth of CNN
        kernel_size : the size of the filter window of n x n
        filter_size : the factor with which to decrease the size: 2 means halving
            the size at each CNN layer and doubling the size at each deconvolution
        pooling_size : pooling windows size of n x n
        n_of_initial_channels : the number of initial channels for the Input. This number
            will be multiplied by 'filter_size' at each consecutive CNN layer
            the deconvolution will 'undo' this by dividing the number of channels 
            at each consecutive tranpose convolution eventually reaching 'n_of_initial_channels'
        func : the activation function. Default is Leaky ReLU
        batch_normalization : as the name suggests. The default is on
        exp_time : probably exposure time
        exp : the number to be used if exp_time is True
    returns:
        Model : a model instance which returns a model with input shape provided above
                the output shape should be the same
    """ 
    convs = []
    pools = []
    input_tensor = Input(shape=input_shape)
    x = input_tensor
    
    for _ in range(depth):
        conv, pool = conv_batch_maxpooling(x, n_of_initial_channels, kernel_size, 
                          pooling_size, func, batch_normalization, use_bias)
        convs.append(conv)
        pools.append(pool)
        x = pool
        n_of_initial_channels *= filter_size
    
    x = convs.pop()
    for i in range(depth - 1):
        exp_ = False
        if i == 0 and exp and exp_time:
            exp_ = True

        x = upsample_and_concat(input_layer=x, previous_layer=convs.pop(), kernel_size=kernel_size, 
                                filter_size=filter_size, func=func, batch_normalization=batch_normalization,
                                 exp=exp_, exp_time=exp_time, use_bias=use_bias)
        
    output_tensor = Conv2D(1, [1, 1], activation=None)(x)
    model = Model(inputs=input_tensor, outputs=output_tensor)
    return model

#These are just to investigate the model
#so far it looks good. The model has to be trained though
depth = 5
kernel_size = 3
filter_size = 2
pooling_size = 2
n_of_initial_channels = 32
input_shape = (128, 128, 3)
model =  network(input_shape, depth, kernel_size, filter_size, pooling_size, n_of_initial_channels,
            func=LeakyReLU, batch_normalization=True, exp_time=None, exp=False)

model.summary()


