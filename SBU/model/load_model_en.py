import numpy as np
from numpy import random as rnd
import keras.optimizers as optimizers
from input_data import *
from scipy import linalg as La
from keras.models import load_model, Model
from model import *
import keras
import os
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


print('started')

SBU_dir = '/home/dyh/Sources/dataset/SBU/'

lr = .001
C = 2 ** (-30)
s = 0.8
N2 = 1  # Number of windows of feature nodes
N3 = 0  # Number of enhancement nodes
epochs = 60  # Number of epochs
adam = optimizers.Adam(lr)
train_pred = []
test_pred = []

X_0, X_1, X_2, X_3, Y = read_train_data(SBU_dir)
X_TEST_0, X_TEST_1, X_TEST_2, X_TEST_3, Y_TEST = read_test_data(SBU_dir)

model_dir = 'train_model/'
#delete_model(model_dir)



model = load_model('/home/dyh/Sources/results/hcn_trained/sbu_test_4_model.h5')
train_pred = model.predict([X_0, X_1, X_2, X_3])
flatten_layer_model = Model(inputs=model.input, outputs=model.get_layer('leaky_re_lu_5').output)
#flatten_layer_model.save('train_model/flatten_layer_model.h5')
en_list = []
test_acc_list = []
test_time_list = []

for en in range(epochs):
    
    N3 = N3 + 10
    FeaturesPerWindow = []
    train_start_time = time.time()
    for i in range(N2):
        FeaturesExtract = flatten_layer_model.predict([X_0, X_1, X_2, X_3])
        print(FeaturesExtract.shape)
        FeaturesPerWindow.append(FeaturesExtract)

    N1 = FeaturesPerWindow[0].shape[1]
    FeaturesPerWindow = np.array(FeaturesPerWindow)
    FeaturesPerWindow = np.reshape(FeaturesPerWindow, newshape=[-1, FeaturesPerWindow[0].shape[1] * N2])

    '''Enhancement Nodes'''
    Features_bias = np.hstack([FeaturesPerWindow, 0.1 * np.ones(((FeaturesPerWindow.shape[0]), 1))])

    if N1 * N2 >= N3:
        random.seed(67797325)
        wh = La.orth(2 * rnd.randn(N2 * N1 + 1, N3)) - 1
    else:
        random.seed(67797325)
        wh = La.orth((2 * rnd.randn(N2 * N1 + 1, N3).T) - 1).T
    print(wh.shape)
    EnhancementFeature = np.dot(Features_bias, wh)
    print(EnhancementFeature.shape)
    l2 = s / np.max(EnhancementFeature)
    EnhancementFeature = tansig(EnhancementFeature * l2)

    FeatureLayerOutput = np.hstack([FeaturesPerWindow, EnhancementFeature])
    beta = pinv(FeatureLayerOutput, C)
    print(FeatureLayerOutput.shape)
    print(beta.shape)
    '''
    rows, cols = FeatureLayerOutput.shape
    l_1 = np.zeros([cols, cols])
    l_2 = np.zeros([cols, cols])
    l_3 = np.zeros([cols])
    swap = np.zeros([cols])
    temp = np.zeros([cols, cols])
    product_I_list = np.zeros([cols, cols])
    beta = np.zeros([cols, rows])
    A_T = np.zeros([cols, rows])
    product = np.zeros([cols, cols])
    product_I = np.zeros([product_I_list.shape[0], product_I_list.shape[1]])
    reg_e = C * np.eye(cols, dtype='float64')
    s = np.zeros([cols, cols])
    
    pinv_matrix(FeatureLayerOutput, s, A_T, product, product_I, beta, product_I_list, l_1, l_2, l_3, swap, temp, reg_e)
    '''

    OutputWeights = np.dot(beta, Y)
    #print(OutputWeights.shape)
    xx = np.dot(FeatureLayerOutput, OutputWeights)
    TrainingAccuracy = show_accuracy(xx, Y)

    print('step: ', en)
    train_end_time = time.time()
    train_time = train_end_time - train_start_time
    print('Training time is' + str(train_time) + 's')
    print('Training accurate is', TrainingAccuracy * 100, '%')



    '''Testing Process'''
    test_start_time = time.time()
    TestFeatures = []
    for j in range(N2):
    #    model_dir = 'train_model/flatten_layer_model.h5'
    #    test_model = load_model(model_dir)
        test_pred = flatten_layer_model.predict([X_TEST_0, X_TEST_1, X_TEST_2, X_TEST_3])
        TestFeatures.append(test_pred)

    TestFeatures = np.array(TestFeatures)
    TestFeatures = np.reshape(TestFeatures, newshape=[-1, TestFeatures[0].shape[1] * N2])
    print(TestFeatures.shape)

    '''Test Enhancement Nodes'''
    TestFeaturesBias = np.hstack([TestFeatures, 0.1 * np.ones(((TestFeatures.shape[0]), 1))])
    TestEnhancementFeatures = np.dot(TestFeaturesBias, wh)
    TestEnhancementFeatures = tansig(TestEnhancementFeatures * l2)

    FinalFeatures = np.hstack([TestFeatures, TestEnhancementFeatures])
    TestOutput = np.dot(FinalFeatures, OutputWeights)
    TestingAccuracy = show_accuracy(TestOutput, Y_TEST)
    test_end_time = time.time()
    test_time = test_end_time - test_start_time
    print('Testing time is' + str(test_time) + 's')
    print('Testing accurate is', TestingAccuracy * 100, '%')
    en_list.append(N3)
    test_time_list.append(test_time)
    test_acc_list.append(TestingAccuracy)

np.savetxt('params/enhance_550.txt', en_list)
np.savetxt('params/test_time_550.txt', test_time_list)
np.savetxt('params/test_acc_550.txt', test_acc_list)
np.savetxt('params/pred_550.txt', TestOutput)
np.savetxt('params/label_550.txt', Y_TEST)


