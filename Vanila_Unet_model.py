import tensorflow as tf
from keras.layers import *
from keras.models import Model
from keras.optimizers import *
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

class Vanila_Unet():
    def __init__(self) -> None:
        self.shape = (256, 1600, 1)
        self.weights = 'model/DefectDetection_dice.h5'

    def dice_coef(self, y_true, y_pred, smooth=1):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    def dice_loss(self, y_true, y_pred):
        smooth = 1.
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = y_true_f * y_pred_f
        score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return 1. - score

    def bce_dice_loss(self, y_true, y_pred):
        return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

    def model_gen(self): 
        input_shape = self.shape
        inputs = Input(input_shape)

        c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (inputs)
        c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
        p1 = MaxPooling2D((2, 2)) (c1)

        c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
        c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
        p2 = MaxPooling2D((2, 2)) (c2)

        c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
        c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
        p3 = MaxPooling2D((2, 2)) (c3)

        c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
        c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
        p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

        c5 = Conv2D(64, (3, 3), activation='relu', padding='same') (p4)
        c5 = Conv2D(64, (3, 3), activation='relu', padding='same') (c5)
        p5 = MaxPooling2D(pool_size=(2, 2)) (c5)

        c55 = Conv2D(128, (3, 3), activation='relu', padding='same') (p5)
        c55 = Conv2D(128, (3, 3), activation='relu', padding='same') (c55)

        u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c55)
        u6 = concatenate([u6, c5])
        c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
        c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)

        u71 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
        u71 = concatenate([u71, c4])
        c71 = Conv2D(32, (3, 3), activation='relu', padding='same') (u71)
        c61 = Conv2D(32, (3, 3), activation='relu', padding='same') (c71)

        u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c61)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
        c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

        u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
        c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

        u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
        c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

        outputs = Conv2D(4, (1, 1), activation='sigmoid') (c9)

        model = Model(inputs=[inputs], outputs=[outputs])
        model.compile(optimizer = 'adam', loss = self.bce_dice_loss, metrics = [self.dice_coef])
        model.load_weights(self.weights)
        return model
if __name__ == '__main__':
    unet = Vanila_Unet()
    model = unet.model_gen()
    print(model.summary())
 
