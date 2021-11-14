image_path = '/home/feng/workspace/self2noise/s9.tif' # 1000x512x512 experimental dataset
training_index = 's9' # the identity for the training, can be a random string
model_directory = f'/home/feng/workspace/self2noise/model_noisy_to_clean_model_{training_index}' # saving the model
threshold = 1 # bottleneck threshold
traing_images = 900 # images used to train the model, the more the better, but the training will be slow
dim = 512 # cut the input images to small pieces for a larger batch size, as normalization layers prefer large batch size
batch_size = 16 # traing batch size, need adjusting according to the GPU memory, 16-32 is good enough
gpu_id = 0 # the GPU to use
zoom_factor = 1 # zoom images before denoising, giving better performance when the shared structure are small in pixels
enhance_contrast = False # enhance contrast
n_loops = 200 # training loops
check_intervals = 4 # the training intervals to see the training result
test_image_index = 0 # the image used to visually evaluate the model's performance in the real time
lr = 0.001 # learing rate
loss = 'mae' # mae performs better than mse when signal is sparse
private_key = None # your private key for Telegram bot. If set to 'None', you will not receive training messages from the bot.
private_id = None # your private id for Telegram bot. If set to 'None', you will not receive training memssages.

from math import exp, sqrt
from scipy.ndimage import zoom
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.layers import LeakyReLU, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import Subtract
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.utils import plot_model
import copy
import glob
import imageio
import numpy as np
import os
import tifffile
import tqdm
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_id)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
config = tf.config.experimental.set_memory_growth(physical_devices[gpu_id], True) # not occupying all the GPU memory

# https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/layers/normalization/instancenormalization.py
class InstanceNormalization(Layer):
    def __init__(self,
                 axis=None,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


from tensorflow.python.keras.utils.generic_utils import get_custom_objects

get_custom_objects().update({'InstanceNormalization': InstanceNormalization})


#
# sending message to telegram
#



def send_message( message ):
    global private_key
    global private_id

    if private_key is None or private_id is None:
        return 0

    command = f'/usr/bin/curl -s -X POST https://api.telegram.org/{private_key}/sendMessage -d chat_id={private_id} -d text="{message}"'
    os.system( command )
    print( '*'*10 )

def send_photo( photo_path ):
    global private_key
    global private_id

    if private_key is None or private_id is None:
        return 0

    command = f'/usr/bin/curl -s -X POST https://api.telegram.org/{private_key}/sendPhoto -F chat_id={private_id} -F photo="@{photo_path}"'
    os.system( command )
    print( '*'*10 )





log_flag = True

def log( message ):
    if log_flag:
        print( message )

'''
    Example:
        model = ...
        write_model('./cached_folder', model)
'''
def write_model(directory, model):
    if not os.path.exists(directory):
        os.makedirs(directory)

    # saving weights
    weights_path = f'{directory}/weights.h5'
    if os.path.isfile(weights_path):
        os.remove(weights_path)
    model.save_weights(weights_path)

    # saving json
    json_path = f'{directory}/js.json'
    if os.path.isfile(json_path):
        os.remove(json_path)
    with open( json_path, 'w' ) as js:
        js.write( model.to_json() )

def write_model_checkpoint(directory, model):
    model_path = f'{directory}/model.h5'
    if os.path.isfile(model_path):
        os.remove(model_path)

    model.save( model_path )
    write_model(directory, model)

'''
    Example:
        model = read_model( './cached_folder' )
'''
def read_model(directory):
    weights_path = f'{directory}/weights.h5'
    if not os.path.isfile(weights_path):
        log( f'No such file {weights_path}' )
        return None

    json_path = f'{directory}/js.json'
    if not os.path.isfile(json_path):
        log( f'No such file {json_path}' )
        return None

    js_file = open( json_path, 'r' )
    model_json = js_file.read()
    js_file.close()
    model = model_from_json( model_json )
    model.load_weights( weights_path )
    return model

def read_model_checkpoint(directory):
    model_path = f'{directory}/model.h5'
    if not os.path.isfile(model_path):
        log( f'No such file {model_path}' )
        return None

    model = load_model(model_path)
    return model

'''
    Example:
        model_a, model_b, ... = generate_model_function(xxx) # <-- weights shared models
        read_weights( './pre_cache_folder', model_a ) #
'''
def read_weights(directory, model):
    weights_path = f'{directory}/weights.h5'

    if not os.path.isfile(weights_path):
        log( f'No such file {weights_path}' )
        return False

    model.load_weights( weights_path )
    return True

layer_counter = 0
def unique_name():
    global layer_counter
    layer_counter += 1
    return 'Layer_'+str(layer_counter).zfill(5)

def make_activation( input_layer, with_normalization=True ):
    if with_normalization:
        return LeakyReLU(alpha=0.2, name=unique_name())(InstanceNormalization(name=unique_name())(input_layer))
    return LeakyReLU(alpha=0.2, name=unique_name())(input_layer)

def make_pooling( input_layer, channels, with_normalization=True ):
    x = conv2d_transpose( channels, kernel_size=(3,3), activation='linear', strides=1, padding='valid')( input_layer )
    x = make_activation( x, with_normalization )
    x = conv2d( channels, kernel_size=(3,3), activation='linear', strides=2, padding='valid')( x )
    x = make_activation( x, with_normalization )
    return x

def conv2d_transpose( *args,**kwargs ):
    if 'name' in kwargs:
        return Conv2DTranspose( *args, **kwargs )
    return Conv2DTranspose( *args, **kwargs, name=unique_name() )

def conv2d( *args,**kwargs ):
    if 'name' in kwargs:
        return Conv2D( *args, **kwargs )
    return Conv2D( *args, **kwargs, name=unique_name() )

def make_block( input_layer, channels, kernel_size=(3,3), with_normalization=True ):
    x = input_layer
    x = conv2d_transpose( channels, kernel_size=kernel_size, activation='linear', strides=1, padding='valid')( x )
    x = make_activation( x, with_normalization )
    x = conv2d( channels, kernel_size=kernel_size, activation='linear', strides=1, padding='valid')( x )
    x = make_activation( x, with_normalization )
    return x

def make_output_block( input_layer, output_channels, kernel_size, output_activation ):
    channels = output_channels << 3
    x = input_layer
    x = conv2d_transpose( channels, kernel_size=kernel_size, activation='linear', strides=1, padding='valid')( x )
    x = make_activation( x )
    x = conv2d( output_channels, kernel_size=kernel_size, activation=output_activation, strides=1, padding='valid')( x )
    return x

def make_upsampling( input_layer, channels ):
    x = conv2d_transpose( channels, kernel_size=(4,4), activation='linear', strides=2, padding='valid')( input_layer )
    x = make_activation( x )
    x = conv2d( channels, kernel_size=(3,3), activation='linear', strides=1, padding='valid')( x )
    x = make_activation( x )
    return x

def make_xception_blocks( input_layer, channels, kernel_sizes ):
    sub_channels = int( channels/len(kernel_sizes) )
    assert sub_channels * len(kernel_sizes) == channels, 'sub-channels and channels not match, adjust the channels or the size of sub-kernels'
    layer_blocks = []
    for kernel_size in kernel_sizes:
        layer_blocks.append( make_block( input_layer, sub_channels, kernel_size ) )
    return concatenate( layer_blocks )

def add( layers ):
    return Add(name=unique_name())( layers )

def make_blocks( n_blocks, input_layer, channels, kernel_size=(3,3) ):
    x = make_block( input_layer, channels, kernel_size )
    for idx in range( n_blocks ):
        x_ = make_block( x, channels, kernel_size )
        x = add( [x_, x] )
    return x

def make_model( input_channels=1, output_channels=1, transform_repeater=16, output_activation='sigmoid', threshold = 16, name=None ):
    input_layer = Input( shape=(None, None, input_channels) )

    gt_128 = input_layer
    gt_64 = AveragePooling2D()( gt_128 )
    gt_32 = AveragePooling2D()( gt_64 )
    gt_16 = AveragePooling2D()( gt_32 )

    encoder_128 = make_xception_blocks( make_block( input_layer, 8, with_normalization=False ), 8, (1, 3, 5, 7) )
    encoder_64 = make_xception_blocks( make_block( make_pooling( encoder_128, 16 ), 16 ), 16, (3, 5) )
    encoder_32 = make_xception_blocks( make_block( make_pooling( encoder_64, 32 ), 32 ), 32, (3, 5) )
    encoder_16 = make_xception_blocks( make_block( make_pooling( encoder_32, 64 ), 64 ), 64, (3, 5) )
    encoder_8  = make_xception_blocks( make_block( make_pooling( encoder_16, 64 ), 64 ), 64, (3, 5) )
    encoder_4  = make_xception_blocks( make_block( make_pooling( encoder_8, 64 ), 64 ), 64, (3, 5) )

    encoder_last = make_block( encoder_4, threshold )

    transformer = make_blocks( transform_repeater, encoder_last, 128 )

    decoder_8 = make_xception_blocks( make_block( make_upsampling( transformer, 128 ), 128 ), 128, (3, 5) )
    decoder_16 = make_xception_blocks( make_block( make_upsampling( decoder_8, 64 ), 64 ), 64, (3, 5) )
    decoder_32 = make_xception_blocks( make_block( make_upsampling( decoder_16, 32 ), 32 ), 32, (3, 5) )
    decoder_64 = make_xception_blocks( make_block( make_upsampling( decoder_32, 16 ), 16 ), 16, (3, 5) )
    decoder_128= make_xception_blocks( make_block( make_upsampling( decoder_64, 8 ), 8 ), 8, (3, 5) )

    output_layer_128 = make_output_block( decoder_128, output_channels, (9, 9), output_activation )
    output_layer_64  = make_output_block(  decoder_64, output_channels, (7, 7), output_activation )
    output_layer_32  = make_output_block(  decoder_32, output_channels, (5, 5), output_activation )
    output_layer_16  = make_output_block(  decoder_16, output_channels, (3, 3), output_activation )

    should_be_zero_128 = Subtract()( [gt_128, output_layer_128] )
    should_be_zero_64  = Subtract()( [gt_64,  output_layer_64 ] )
    should_be_zero_32  = Subtract()( [gt_32,  output_layer_32 ] )
    should_be_zero_16  = Subtract()( [gt_16,  output_layer_16 ] )

    mcnn_model = Model( input_layer, [should_be_zero_128, should_be_zero_64, should_be_zero_32, should_be_zero_16], name='mcnn_model' )
    self2noise_model = Model( input_layer, output_layer_128, name='self2noise_model' )

    return mcnn_model, self2noise_model



# preparing model path
if not os.path.exists(model_directory):
    os.mkdir(model_directory)


# creating model
send_message( f'self2noise training {training_index} started' )
mcnn, s2n = make_model(threshold=threshold)
mcnn.compile(loss=loss, optimizer=Adam(learning_rate=lr))
experimental_images = tifffile.imread( image_path )
noisy_images_ = experimental_images


# preprocessing traing images
n, row, col = noisy_images_.shape
if n > traing_images:
    noisy_images_ = noisy_images_[:traing_images]
    n, row, col = noisy_images_.shape
if zoom_factor > 1:
    u_noisy_images_ = []
    for idx in range( n ):
        x = zoom( noisy_images_[idx], zoom_factor, order=3 )
        u_noisy_images_.append( x )
    noisy_images_ = np.asarray( u_noisy_images_ )
noisy_images_ = noisy_images_ * 1.0
noisy_images_ /= np.amax( noisy_images_ ) + 1.0e-10
send_message( f'noisy image of shape {noisy_images_.shape} loaded' )


# preprocessing noisy dataset for the training
factor = int( row / dim )
noisy_images = np.zeros( (n*factor*factor, dim, dim), dtype=noisy_images_.dtype )
for r in range( factor ):
    for c in range( factor ):
        offset = r * factor + c
        noisy_images[offset*n:(1+offset)*n] = noisy_images_[:,r*dim:(r+1)*dim, c*dim:(c+1)*dim]
noisy_images = noisy_images.reshape( noisy_images.shape + (1,) )
n_noisy_images, *_ = noisy_images.shape
if enhance_contrast :
    noisy_images *= noisy_images



# the mcnn output
zeros = [np.zeros((batch_size, dim, dim, 1)), np.zeros((batch_size, dim//2, dim//2, 1)), np.zeros((batch_size, dim//4, dim//4, 1)), np.zeros((batch_size, dim//8, dim//8, 1))]


# the image to test on
test_image = np.squeeze( noisy_images_[test_image_index] )
row, col = test_image.shape
n_pixels = 2048
if row > n_pixels and col > n_pixels:
    test_image = test_image[((row-n_pixels)>>1):((row-n_pixels)>>1) + n_pixels, ((col-n_pixels)>>1):((col-n_pixels)>>1)+n_pixels]
imageio.imsave( './self2noise_test.png', np.squeeze(test_image) )
send_photo( './self2noise_test.png' )
test_image = test_image.reshape( (1,) + test_image.shape + (1,) )
test_image = test_image / (np.amax(test_image) + 1.0e-10)


# starting training
current_losses = None
for loop in range( n_loops ):

    for idx in range( int(n_noisy_images/batch_size) ):
        input = noisy_images[idx*batch_size:(idx+1)*batch_size]
        outputs = zeros
        current_losses = mcnn.train_on_batch( input, outputs )
        print( f'self2noise --> {loop}/{n_loops} with minibatch {idx*batch_size}/{n_noisy_images}, losses: {current_losses}', end='\r')

    if (loop != 0 and (loop % check_intervals) == 0):
        translated_image = np.squeeze( s2n.predict( test_image ) )
        imageio.imsave( f'./self2noise_denoised_{loop}.png', np.squeeze(translated_image) )
        send_message( f'self2noise denoising of {training_index}: {loop+1}/{n_loops} done, last loss is {current_losses}.' )
        send_photo( f'./self2noise_denoised_{loop}.png' )
        write_model( f'{model_directory}/final_model', s2n )


# denoising on the whole dataset
if zoom_factor > 1:
    u_noisy_images_ = []
    n, *_ = experimental_images.shape
    for idx in range( n ):
        x = zoom( experimental_images[idx], zoom_factor, order=3 )
        u_noisy_images_.append( x )
    experimental_images = np.asarray( u_noisy_images_ )
experimental_images = 1.0 * experimental_images
experimental_images -= np.amin(experimental_images)
experimental_images /= np.amax(experimental_images)
if enhance_contrast:
    experimental_images *= experimental_images
experimental_images = experimental_images.reshape( experimental_images.shape + (1,) )
denoised_images = s2n.predict( experimental_images, batch_size=1 )
write_model( f'{model_directory}/final_model', s2n )
tifffile.imwrite( f'{image_path}_denoised.tif', np.asarray( np.squeeze(denoised_images)*65535.0, dtype='uint16' ) )
send_message( f'>> self2noise denoising of {training_index} finished.' )

