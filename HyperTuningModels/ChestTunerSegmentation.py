""" Keras Tuner Chest Segmentation """ 

import kerastuner as kt
import tensorflow as tf
import keras
import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import backend
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from kerastuner.engine import hypermodel
from zipfile import ZipFile

from kerastuner.tuners import Hyperband

from utils import *
import tqdm
from sklearn.model_selection import train_test_split
import numpy as np


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

from kerastuner.engine import hypermodel


class HyperUNet(hypermodel.HyperModel):
    """A UNet HyperModel.
    # Arguments:
        include_top: whether to include the fully-connected
            layer at the top of the network.
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

        super(HyperUNet, self).__init__(**kwargs)
        
        
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
        activation = hp.Choice('activation', values = ['relu', 'prelu'])
        use_batch_norm = hp.Boolean('batch_norm', default = False)
        dropout = hp.Float('dropout', 0, 0.5, 0.05, default = 0)
        dropout_change_per_layer = hp.Float('drop_out_change_per_layer', 0.0,0.3,0.025)
        use_dropout_on_upsampling = hp.Boolean('dropout_on_upsampling', default = False)
        decoder_type = hp.Choice('decoder_type', values = ['simple', 'transpose'])
        
        #hyperparameters not tuning
        dropout_type="standard"
        output_activation="sigmoid"
        
        # Build U-Net model
        inputs = Input(input_shape)
        x = inputs

        down_layers = []
        for l in range(num_layers):
            x = conv2d_block(
            inputs=x,
            filters=filters,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            dropout_type=dropout_type,
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
            dropout_type=dropout_type,
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
                x = UpSampling2D(filters, (2, 2), strides=(2, 2), padding="same")(x)
                x = concatenate([x, conv])
                x = conv2d_block(
                inputs=x,
                filters=filters,
                use_batch_norm=use_batch_norm,
                dropout=dropout,
                dropout_type=dropout_type,
                activation=activation,
            )

        outputs = Conv2D(num_classes, (1, 1), activation=output_activation)(x)

        
        #model = Model(inputs=[inputs], outputs=[outputs])
        
        model = keras.Model(inputs, outputs, name='UNet')
        optimizer_name = hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd'], default='adam')
        optimizer = keras.optimizers.get(optimizer_name)
        optimizer.learning_rate = hp.Choice(
                'learning_rate', [0.1, 0.01, 0.001], default=0.01)
        model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy'])
        return model
    
def conv2d_block(
        inputs,
        use_batch_norm=True,
        dropout=0.3,
        dropout_type="spatial",
        filters=16,
        kernel_size=(3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same"):

        if dropout_type == "spatial":
            DO = SpatialDropout2D
        elif dropout_type == "standard":
            DO = layers.Dropout
        else:
            raise ValueError(
                f"dropout_type must be one of ['spatial', 'standard'], got {dropout_type}"
            )

        c = layers.Conv2D(filters,kernel_size,activation=activation,
        kernel_initializer=kernel_initializer,
        padding=padding,use_bias=not use_batch_norm)(inputs)
        if use_batch_norm:
            c = BatchNormalization()(c)
        if dropout > 0.0:
            c = DO(dropout)(c)
        c = layers.Conv2D(filters,kernel_size,activation=activation,
            kernel_initializer=kernel_initializer,padding=padding,use_bias=not use_batch_norm)(c)
        if use_batch_norm:
            c = layers.BatchNormalization()(c)
        return c



#import data for tuning the model 

image_paths,mask_paths = load_lung_images("Data/lgg-mri-segmentation/kaggle_3m/")

masks,images = read_in_images(image_paths,mask_paths)

images = images[0:400]
masks = masks[0:400]
    
for i in range(0,len(masks)):
    m = masks[i]
    m = m[:,:,0]
    m.reshape(m.shape[0],m.shape[1])
    masks[i] = m

images = np.asarray(images)
masks = np.asarray(masks)

#split into validations
img_overall_train, img_test, mask_overall_train, mask_test = train_test_split(images, masks, test_size=0.16667, random_state=42)

train_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(
    img_overall_train, mask_overall_train,
    batch_size=32)

val_generator = train_datagen.flow(
    img_test, mask_test)
   


INPUT_SHAPE = images[0].shape
NUM_CLASSES = 1
HYPERBAND_MAX_EPOCHS = 22
SEED = 2

hypermodel = HyperUNet(input_shape = INPUT_SHAPE, classes=NUM_CLASSES)


tuner = Hyperband(
    hypermodel,
    max_epochs=HYPERBAND_MAX_EPOCHS,
    objective='val_accuracy',
    seed=SEED,
    executions_per_trial=22)


tuner.search(train_generator, epochs=20, validation_data = val_generator)

best_model = tuner.get_best_models(1)[0]

best_hyperparameters = tuner.get_best_hyperparameters(1)[0]


#
model_json = best_model.to_json()
with open("LGG_basicUNet_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
best_model.save_weights("LGG_basicUNet_tuner_model.h5")

with open('best_LGG_basicUNet_Param.txt', 'w') as f:
    print(best_hyperparameters, file=f)


print("Saved model to disk")