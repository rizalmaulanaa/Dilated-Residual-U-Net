# modified code from https://github.com/DuFanXin/deep_residual_unet/blob/master/res_unet.py
from tensorflow.keras.layers import UpSampling2D, Conv2D, Add
from tensorflow.keras.layers import BatchNormalization, Concatenate, Activation

def DilatedBlock(filter):
    def layer(input_tensor):
        dilation_rates = [1,2,4,8]
        x = [None]*(len(dilation_rates)+1)
        x[0] = input_tensor

        for k,i in enumerate(dilation_rates):
            x[k+1] = Conv2D(filter, kernel_size=3, dilation_rate=i, padding='same',
                            name='dilation_block_'+str(i)) (x[k])

        x = Add(name='dilation_block_add') (x[1:])
        return x
    return layer

def res_block(x, nb_filters, strides):
    res_path = BatchNormalization()(x)
    res_path = Activation(activation='relu')(res_path)
    res_path = Conv2D(filters=nb_filters[0], kernel_size=(3, 3), padding='same', strides=strides[0])(res_path)
    res_path = BatchNormalization()(res_path)
    res_path = Activation(activation='relu')(res_path)
    res_path = Conv2D(filters=nb_filters[1], kernel_size=(3, 3), padding='same', strides=strides[1])(res_path)

    shortcut = Conv2D(nb_filters[1], kernel_size=(1, 1), strides=strides[0])(x)
    shortcut = BatchNormalization()(shortcut)

    res_path = Add() ([shortcut, res_path])
    return res_path


def encoder(x, num_filters=[64, 128, 256]):
    to_decoder = []
    main_path = x
    for k,i in enumerate(num_filters):
        if k == 0:
            main_path = Conv2D(filters=i, kernel_size=(3, 3), padding='same', strides=(1, 1))(main_path)
            main_path = BatchNormalization()(main_path)
            main_path = Activation(activation='relu')(main_path)

            main_path = Conv2D(filters=i, kernel_size=(3, 3), padding='same', strides=(1, 1))(main_path)

            shortcut = Conv2D(filters=i, kernel_size=(1, 1), strides=(1, 1))(x)
            shortcut = BatchNormalization()(shortcut)

            main_path = Add() ([shortcut, main_path])

        else:
            main_path = res_block(main_path, [i, i], [(2, 2), (1, 1)])

        # branching to decoder
        to_decoder.append(main_path)

    return to_decoder


def decoder(x, from_encoder, num_filters=[64, 128, 256]):
    main_path = x
    for k, i in reversed(list(enumerate(num_filters))):
        main_path = UpSampling2D(size=(2, 2))(main_path)
        main_path = Concatenate(axis=-1) ([main_path, from_encoder[k]])
        main_path = res_block(main_path, [i, i], [(1, 1), (1, 1)])

    return main_path
