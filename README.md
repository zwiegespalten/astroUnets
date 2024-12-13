# astroUnets

## create_new_image.py

This is the original network put forward by @Sponka at https://github.com/Sponka/Astro_U-net/blob/master/code/create_new_image.py#L20
However, the original network has been written in Tensorflow.1 which is outdated.

## network.ipyb

This notebook recreates the original network provided above with some improvements and changes

1) An optional 'Batch Normalization' as well as 'Bias Term' have been added after each convolution. The default is batch 'normalization=True' and 'bias_on=False'
2) Sponka's original U-Net has had a hardcoded 'Leaky ReLU' after each convolution. While this is still the default, another activation function can be provided via the paramemeter 'func='
3) The original network consisted of 5 layer where each layer was made up of 2 convolutions followed up by a  max pooling. The layer struction has been kept. However, the depth of the network can now be provided via the parameter 'depth'
