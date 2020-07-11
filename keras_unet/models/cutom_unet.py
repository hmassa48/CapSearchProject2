from keras.models import Model
from keras.layers import (
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
        Activation)



def conv2d_block(
    inputs,
    use_batch_norm=True,
    dropout=0.3,
    dropout_type="standard",
    filters=16,
    kernel_size=(3, 3),
    activation="relu",
    kernel_initializer="he_normal",
    padding="same",
):

    if dropout_type == "spatial":
        DO = SpatialDropout2D
    elif dropout_type == "standard":
        DO = Dropout
    else:
        raise ValueError(
            f"dropout_type must be one of ['spatial', 'standard'], got {dropout_type}"
        )

    c = Conv2D(
        filters,
        kernel_size,
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=not use_batch_norm,
    )(inputs)
    if use_batch_norm:
        c = BatchNormalization()(c)
    if dropout > 0.0:
        c = DO(dropout)(c)
    c = Conv2D(
        filters,
        kernel_size,
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=not use_batch_norm,
    )(c)
    if use_batch_norm:
        c = BatchNormalization()(c)
    return c


def custom_unet(
    input_shape,
    num_classes=1,
    activation="relu",
    use_batch_norm=True,
    decoder_type = 'upsample',  # 'deconv' or 'simple'
    dropout=0.3,
    dropout_change_per_layer=0.0,
    dropout_type="standard",
    use_dropout_on_upsampling=False,
    use_attention=False,
    filters=16,
    num_layers=4,
    output_activation="sigmoid",
):  # 'sigmoid' or 'softmax'

    """
    Customisable UNet architecture (Ronneberger et al. 2015).
    Cutomization inspired by the U-Net ++ architecture. This allows for decoder blocks to be traditional UpSampling Blocks or Transpose blocks. 

    """

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

    model = Model(inputs=[inputs], outputs=[outputs])
    return model
