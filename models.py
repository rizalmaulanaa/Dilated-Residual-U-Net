# modified code from https://github.com/DuFanXin/deep_residual_unet/blob/master/res_unet.py
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPool2D

from blocks import encoder, decoder, res_block, DilatedBlock

def D_Residual_Unet(input_shape, dilated_block=True):
    inputs = Input(shape=input_shape)
    to_decoder = encoder(inputs)
    if dilated_block:
        path = MaxPool2D(pool_size=2) (to_decoder[-1])
        path = DilatedBlock(512) (path)
    else:
        path = res_block(to_decoder[-1], [512, 512], [(2, 2), (1, 1)])

    path = decoder(path, from_encoder=to_decoder)
    path = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid') (path)

    return Model(inputs=inputs, outputs=path, name='Dilated_Residual_U_Net')
