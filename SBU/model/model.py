from __future__ import division
from keras.models import Model
from keras.layers import *
from keras.layers.core import *
from keras.layers.convolutional import *
#import numba


def one_obj(frame_l=16, joint_n=15, joint_d=3):
    input_joints = Input(name='joints', shape=(frame_l, joint_n, joint_d))
    input_joints_diff = Input(name='joints_diff', shape=(frame_l, joint_n, joint_d))

    ##########branch 1##############

    # Conv1-layer1
    x = Conv2D(name='sp_conv1_layer1', filters=32, kernel_size=(1, 1), padding='same')(input_joints)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # Conv2-layer2
    x = Conv2D(name='sp_conv2_layer2', filters=16, kernel_size=(3, 1), padding='same')(x)
    x = BatchNormalization()(x)
    # x = LeakyReLU()(x)

    x = Permute((1, 3, 2))(x)

    # Conv3-layer3
    x = Conv2D(name='sp_conv3_layer3', filters=16, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    # x = LeakyReLU()(x)
    x = MaxPooling2D(name='sp_pool1_layer3', pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    # Conv4-layer4
    x = Conv2D(name='sp_conv4_layer4', filters=64, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    # x = LeakyReLU()(x)
    x = MaxPooling2D(name='sp_pool2_layer4', pool_size=(3, 3), strides=(2, 2), padding='same')(x)


    ##########branch 2##############Temporal difference

    # Conv1-layer1
    x_d = Conv2D(name='tem_conv1_layer1', filters=32, kernel_size=(1, 1), padding='same')(input_joints_diff)
    x_d = BatchNormalization()(x_d)
    x_d = LeakyReLU()(x_d)

    # Conv2-layer2
    x_d = Conv2D(name='tem_conv2_layer2', filters=16, kernel_size=(3, 1), padding='same')(x_d)
    x_d = BatchNormalization()(x_d)
    # x_d = LeakyReLU()(x_d)

    x_d = Permute((1, 3, 2))(x_d)

    # Conv3-layer3
    x_d = Conv2D(name='tem_conv3_layer3', filters=16, kernel_size=(3, 3), padding='same')(x_d)
    x_d = BatchNormalization()(x_d)
    # x_d = LeakyReLU()(x_d)
    x_d = MaxPooling2D(name='tem_pool1_layer3', pool_size=(3, 3), strides=(2, 2), padding='same')(x_d)

    # Conv4-layer4
    x_d = Conv2D(name='tem_conv4_layer4', filters=64, kernel_size=(3, 3), padding='same')(x_d)
    x_d = BatchNormalization()(x_d)
    # x_d = LeakyReLU()(x_d)
    x_d = MaxPooling2D(name='tem_pool2_layer4', pool_size=(3, 3), strides=(2, 2), padding='same')(x_d)

    x = Flatten()(x)
    x_d = Flatten()(x_d)
    features = concatenate([x, x_d], axis=-1)
    model = Model(inputs=[input_joints, input_joints_diff], outputs=features)

    return model


def multi_obj(frame_l=16, joint_n=15, joint_d=3):
    inp_j_0 = Input(name='inp_j_0', shape=(frame_l, joint_n, joint_d))
    inp_j_diff_0 = Input(name='inp_j_diff_0', shape=(frame_l, joint_n, joint_d))

    inp_j_1 = Input(name='inp_j_1', shape=(frame_l, joint_n, joint_d))
    inp_j_diff_1 = Input(name='inp_j_diff_1', shape=(frame_l, joint_n, joint_d))

    # FeatureNodes = []
    single = one_obj()
    # for i in range(N2):
    x_0 = single([inp_j_0, inp_j_diff_0])
    x_1 = single([inp_j_1, inp_j_diff_1])
    final_feature = concatenate([x_0, x_1])

    model = Model([inp_j_0, inp_j_diff_0, inp_j_1, inp_j_diff_1], final_feature)

    return model


#@cuda.jit
def tansig(x):
    return (2 / (1 + np.exp(-2 * x))) - 1

def tanh(data):
    return (np.exp(data) - np.exp(-data)) / (np.exp(data) + np.exp(-data))


def relu(data):
    return np.maximum(data, 0)


#@numba.jit
def pinv(A, reg):
    #reg = 2 ** (-30)
    return np.mat(reg * np.eye(A.shape[1]) + A.T.dot(A)).I.dot(A.T)


def show_accuracy(predictLabel, Label):
    count = 0
    label_1 = np.zeros(Label.shape[0])
    predlabel = []
    label_1 = Label.argmax(axis=1)
    predlabel = predictLabel.argmax(axis=1)
    for j in list(range(Label.shape[0])):
        if label_1[j] == predlabel[j]:
            count += 1
    return round(count / len(Label), 5)
