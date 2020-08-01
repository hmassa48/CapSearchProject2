""" 
Tunable Hyper-Parameter U-Net. Based on the network created in the customizable U-Net but tunable with the keras-tuner. 
"""

import kerastuner as kt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import backend

from blocks import conv2d_block
from kerastuner.engine import hypermodel
from ..metrics import iou, iou_thresholded

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    Dropout,
    SpatialDropout2D,
    UpSampling2D,
    Input,
    concatenate,
    multiply,
    add,
    Activation,
)



class HyperBasicUNet(hypermodel.HyperModel):
    """A UNet HyperModel.
    # Arguments:
        input_shape: Optional shape tuple, e.g. `(256, 256, 3)`.
              One of `input_shape` or `input_tensor` must be
              specified.
        input_tensor: Optional Keras tensor (i.e. output of
            `layers.Input()`) to use as image input for the model.
              One of `input_shape` or `input_tensor` must be
              specified.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True,
            and if no `weights` argument is specified.
        **kwargs: Additional keyword arguments that apply to all
            HyperModels. See `kerastuner.HyperModel`.
    """

    def __init__(self,
                 #include_top=True,
                 input_shape=None,
                 input_tensor=None,
                 classes=None,
                 **kwargs):

        super(HyperBasicUNet, self).__init__(**kwargs)
        
        
        #if include_top and classes is None:
            #raise ValueError('You must specify `classes` when '
                             #'`include_top=True`')

        if input_shape is None and input_tensor is None:
            raise ValueError('You must specify either `input_shape` '
                             'or `input_tensor`.')

        #self.include_top = include_top
        self.input_shape = input_shape
        self.input_tensor = input_tensor
        self.classes = classes
        
        
    def build(self, hp):
        num_layers = hp.Int('layers', 3,6,1, default = 4)
        filters = hp.Choice('filters', values = [16,32,64,128], default = 64)
        kernel = hp.Choice('kernel', values = [3,5,7], default = 3)
        activation = hp.Choice('activation', values = ['relu', 'elu'])
        use_batch_norm = hp.Boolean('batch_norm', default = False)
        dropout = hp.Float('dropout', 0, 0.4, 0.05, default = 0)
        dropout_change_per_layer = hp.Float('drop_out_change_per_layer', 0.0,0.3,0.025)
        use_dropout_on_upsampling = hp.Boolean('dropout_on_upsampling', default = False)
        decoder_type = hp.Choice('decoder_type', values = ['simple', 'transpose'])
        
        #hyperparameters not tuning
        output_activation="sigmoid"
        strides = (2,2)

        
        # Build U-Net model
        inputs = Input(self.input_shape)
        x = inputs

        down_layers = []
        for l in range(num_layers):
            x = conv2d_block(
            inputs=x,
            filters=filters,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            activation=activation,
            )
            down_layers.append(x)
            x = MaxPooling2D((2, 2))(x)
            dropout += dropout_change_per_layer
            filters = filters * 2  # double the number of filters with each layer

        x = conv2d_block(
            inputs=x,
            filters=filters,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            activation=activation,
        )

        if not use_dropout_on_upsampling:
            dropout = 0.0
            dropout_change_per_layer = 0.0

        for conv in reversed(down_layers):
            filters //= 2  # decreasing number of filters with each layer
            dropout -= dropout_change_per_layer
            if decoder_type == 'transpose':
                x = Conv2DTranspose(filters, (3,3), strides=(2,2),
                            padding='same', use_bias=not(use_batch_norm))(x)
                if use_batch_norm:
                    x = BatchNormalization()(x)
                x = Activation(activation)(x)
                x = concatenate([x, conv])
                x = Conv2D(filters,(3,3),activation=activation,kernel_initializer='he_normal',padding='same',use_bias=not use_batch_norm)
                if use_batch_norm:
                    x = BatchNormalization()(x)
                x = Activation(activation)(x)
                
            else:
                if decoder_type == 'simple_bilinear':
                    x = Conv2D(filters, 2, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(interpolation='bilinear')(x))
                else:
                    x = Conv2D(filters, 2, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(strides)(x))
                x = concatenate([x, conv])
                x = conv2d_block(
                    inputs=x,
                    filters=filters,
                    use_batch_norm=use_batch_norm,
                    dropout=dropout,
                    activation=activation,
            )

        outputs = Conv2D(self.classes, (1, 1), activation=output_activation)(x)

        
        #model = Model(inputs=[inputs], outputs=[outputs])
        
        model = keras.Model(inputs, outputs, name='UNet')
        optimizer_name = hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd'], default='adam')
        optimizer = keras.optimizers.get(optimizer_name)
        optimizer.learning_rate = hp.Choice(
                'learning_rate', [0.1, 0.01, 0.001], default=0.01)
        model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=[iou, iou_thresholded])
        return model
    
