import numpy as np
import scipy.io as scio
from sklearn import preprocessing
from scipy import linalg as La
from numpy import random
import time


def shrinkage(a, b):
    z = np.maximum(a - b, 0) - np.maximum(-a - b, 0)
    return z


def sparse_bls(A, b, lam, itrs):
    AA = A.T.dot(A)
    m = A.shape[1]
    n = b.shape[1]
    x = np.zeros((m, n))
    wk = x
    ok = x
    uk = x
    L1 = np.eye(m) / (AA + np.eye(m))
    # L1 = np.mat(AA + np.eye(m)).I
    L2 = (L1.dot(A.T)).dot(b)

    for i in range(itrs):
        tempc = ok - uk;
        ck = L2 + L1.dot(tempc)
        ok = shrinkage(ck + uk, lam)
        uk = uk + (ck - ok)
        wk = ok
    return wk


def tansig(x):
    return (2 / (1 + np.exp(-2 * x))) - 1


def sigmoid(data):
    return 1.0 / (1 + np.exp(-data))


def linear(data):
    return data


def tanh(data):
    return (np.exp(data) - np.exp(-data)) / (np.exp(data) + np.exp(-data))


def relu(data):
    return np.maximum(data, 0)


def pinv(A, reg):
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


def bls_train(train_x, train_y, test_x, test_y, s, c, N1, N2, N3):
    '''
    :param N1: Feature nodes per window
    :param N2: Number of windows of feature nodes
    :param N3: Number of enhancement nodes
    :return:
    '''
    # train_x = preprocessing.scale(train_x, axis=1)
    DataBias = np.hstack([train_x, 0.1 * np.ones(((train_x.shape[0]), 1))])
    OutputOfFeatureMappingLayer = np.zeros((train_x.shape[0], N2 * N1))
    Weights_input2features = []
    distOfMaxAndMin = []
    minOfEachWindow = []
    ymin = 0
    ymax = 1
    L = 0
    train_acc_all = np.zeros([1, L + 1])
    test_acc = np.zeros([1, L + 1])
    train_time = np.zeros([1, L + 1])
    test_time = np.zeros([1, L + 1])
    time_start = time.time()  # 计时开始
    for i in range(N2):
        weight_per_window = 2 * np.random.randn(train_x.shape[1] + 1, N1) - 1
        # weight_per_window = 2 * np.ones([train_x.shape[1] + 1, N1], dtype=float) - 1
        features_per_window = np.dot(DataBias, weight_per_window)
        scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(features_per_window)
        normalized_features = scaler.transform(features_per_window)
        beta_per_window = sparse_bls(normalized_features, DataBias, 1e-3, 50).T
        Weights_input2features.append(beta_per_window)

        feature_nodes_values = np.dot(DataBias, beta_per_window)
        distOfMaxAndMin.append(np.max(feature_nodes_values, axis=0) - np.min(feature_nodes_values, axis=0))
        minOfEachWindow.append(np.min(feature_nodes_values, axis=0))
        feature_nodes_values = (feature_nodes_values - minOfEachWindow[i]) / distOfMaxAndMin[i]
        OutputOfFeatureMappingLayer[:, N1 * i:N1 * (i + 1)] = feature_nodes_values
        del feature_nodes_values
        del features_per_window
        del weight_per_window

    '''Enhancement nodes'''
    features_bias = np.hstack([OutputOfFeatureMappingLayer, 0.1 * np.ones(((OutputOfFeatureMappingLayer.shape[0]), 1))])

    if N1 * N2 >= N3:
        random.seed(67797325)
        wh = La.orth(2 * random.randn(N2 * N1 + 1, N3)) - 1
    else:
        random.seed(67797325)
        wh = La.orth((2 * random.randn(N2 * N1 + 1, N3).T) - 1).T

    enhancement_values = np.dot(features_bias, wh)
    l2 = s / np.max(enhancement_values)
    enhancement_values = tansig(enhancement_values * l2)

    features_enhancements = np.hstack([OutputOfFeatureMappingLayer, enhancement_values])
    del features_bias
    del enhancement_values
    beta = pinv(features_enhancements, c)
    OutputWeights = np.dot(beta, train_y)

    print(OutputWeights.shape)

    xx = np.dot(features_enhancements, OutputWeights)
    TrainingAccuracy = show_accuracy(xx, train_y)
    time_end = time.time()  # 训练完成
    Training_time = time_end - time_start
    print('Training accurate is', TrainingAccuracy * 100, '%')
    print('Training time is ', Training_time, 's')
    train_acc_all[0][0] = TrainingAccuracy
    train_time[0][0] = Training_time

    '''Testing'''
    test_x = preprocessing.scale(test_x, axis=1)
    test_data_bias = np.hstack([test_x, 0.1 * np.ones((test_x.shape[0], 1))])
    OutputOfFeatureMappingLayerTest = np.zeros((test_x.shape[0], N1 * N2))
    time_start = time.time()

    for i in range(N2):
        features_per_window_test = np.dot(test_data_bias, Weights_input2features[i])
        OutputOfFeatureMappingLayerTest[:, N1 * i:N1 * (i + 1)] = (ymax - ymin) * (
                features_per_window_test - minOfEachWindow[i]) / distOfMaxAndMin[i] - ymin

    enhancement_features_bias = np.hstack(
        [OutputOfFeatureMappingLayerTest, 0.1 * np.ones((OutputOfFeatureMappingLayerTest.shape[0], 1))])
    test_enhancement_values = np.dot(enhancement_features_bias, wh)
    test_enhancement_values = tansig(test_enhancement_values * l2)

    final_input = np.hstack([OutputOfFeatureMappingLayerTest, test_enhancement_values])
    Output_test = np.dot(final_input, OutputWeights)
    time_end = time.time()
    Testing_time = time_end - time_start
    TestingAccuracy = show_accuracy(Output_test, test_y)
    print('Testing accurate is', TestingAccuracy * 100, '%')
    print('Testing time is ', Testing_time, 's')

    test_acc[0][0] = TestingAccuracy
    test_time[0][0] = Testing_time
    return TrainingAccuracy, TestingAccuracy, Training_time, Testing_time


def normalize_data(x):
    # column_max = np.max(x, axis=0)
    # column_min = np.min(x, axis=0)
    # nomalized_data = (x - column_min + 1) / (column_max - column_min + 1)
    nomalized_data = x / 255
    return nomalized_data


def BLS_AddEnhanceNodes(train_x, train_y, test_x, test_y, s, c, N1, N2, N3, L, M):
    # 生成映射层
    '''
    两个参数最重要，1）y;2)Beta1OfEachWindow
    '''
    u = 0
    ymax = 1  # 数据收缩上限
    ymin = 0  # 数据收缩下限
    train_x = preprocessing.scale(train_x, axis=1)  # 处理数据
    FeatureOfInputDataWithBias = np.hstack([train_x, 0.1 * np.ones((train_x.shape[0], 1))])
    OutputOfFeatureMappingLayer = np.zeros([train_x.shape[0], N2 * N1])
    #    Beta1OfEachWindow = np.zeros([N2,train_x.shape[1]+1,N1])
    distOfMaxAndMin = []
    minOfEachWindow = []
    train_acc = np.zeros([1, L + 1])
    test_acc = np.zeros([1, L + 1])
    train_time = np.zeros([1, L + 1])
    test_time = np.zeros([1, L + 1])
    time_start = time.time()  # 计时开始
    Beta1OfEachWindow = []
    for i in range(N2):
        random.seed(i + u)
        weightOfEachWindow = 2 * random.randn(train_x.shape[1] + 1, N1) - 1  # 生成每个窗口的权重系数，最后一行为偏差
        #        WeightOfEachWindow([],[],i) = weightOfEachWindow; #存储每个窗口的权重系数
        FeatureOfEachWindow = np.dot(FeatureOfInputDataWithBias, weightOfEachWindow)  # 生成每个窗口的特征
        # 压缩每个窗口特征到[-1，1]
        scaler1 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(FeatureOfEachWindow)
        FeatureOfEachWindowAfterPreprocess = scaler1.transform(FeatureOfEachWindow)
        # 通过稀疏化计算映射层每个窗口内的最终权重
        betaOfEachWindow = sparse_bls(FeatureOfEachWindowAfterPreprocess, FeatureOfInputDataWithBias, 1e-3, 50).T
        Beta1OfEachWindow.append(betaOfEachWindow)
        outputOfEachWindow = np.dot(FeatureOfInputDataWithBias, betaOfEachWindow)
        distOfMaxAndMin.append(np.max(outputOfEachWindow, axis=0) - np.min(outputOfEachWindow, axis=0))
        minOfEachWindow.append(np.min(outputOfEachWindow, axis=0))
        outputOfEachWindow = (outputOfEachWindow - minOfEachWindow[i]) / distOfMaxAndMin[i]
        OutputOfFeatureMappingLayer[:, N1 * i:N1 * (i + 1)] = outputOfEachWindow
        del outputOfEachWindow
        del FeatureOfEachWindow
        del weightOfEachWindow

        # 生成强化层
    # 以下为映射层输出加偏置（强化层输入）
    InputOfEnhanceLayerWithBias = np.hstack(
        [OutputOfFeatureMappingLayer, 0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0], 1))])
    # 生成强化层权重
    if N1 * N2 >= N3:
        random.seed(67797325)
        weightOfEnhanceLayer = La.orth(2 * random.randn(N2 * N1 + 1, N3) - 1)
    else:
        random.seed(67797325)
        weightOfEnhanceLayer = La.orth(2 * random.randn(N2 * N1 + 1, N3).T - 1).T

    tempOfOutputOfEnhanceLayer = np.dot(InputOfEnhanceLayerWithBias, weightOfEnhanceLayer)
    parameterOfShrink = s / np.max(tempOfOutputOfEnhanceLayer)
    OutputOfEnhanceLayer = tansig(tempOfOutputOfEnhanceLayer * parameterOfShrink)

    # 生成最终输入
    InputOfOutputLayer = np.hstack([OutputOfFeatureMappingLayer, OutputOfEnhanceLayer])
    pinvOfInput = pinv(InputOfOutputLayer, c)
    OutputWeight = pinvOfInput.dot(train_y)  # 全局违逆
    time_end = time.time()  # 训练完成
    trainTime = time_end - time_start

    # 训练输出
    OutputOfTrain = np.dot(InputOfOutputLayer, OutputWeight)
    trainAcc = show_accuracy(OutputOfTrain, train_y)
    print('Training accurate is', trainAcc * 100, '%')
    print('Training time is ', trainTime, 's')
    train_acc[0][0] = trainAcc
    train_time[0][0] = trainTime
    # 测试过程
    test_x = preprocessing.scale(test_x, axis=1)  # 处理数据 x = (x-mean(x))/std(x) x属于[-1，1]
    FeatureOfInputDataWithBiasTest = np.hstack([test_x, 0.1 * np.ones((test_x.shape[0], 1))])
    OutputOfFeatureMappingLayerTest = np.zeros([test_x.shape[0], N2 * N1])
    time_start = time.time()  # 测试计时开始
    #  映射层
    for i in range(N2):
        outputOfEachWindowTest = np.dot(FeatureOfInputDataWithBiasTest, Beta1OfEachWindow[i])
        OutputOfFeatureMappingLayerTest[:, N1 * i:N1 * (i + 1)] = (ymax - ymin) * (
                outputOfEachWindowTest - minOfEachWindow[i]) / distOfMaxAndMin[i] - ymin
    #  强化层
    InputOfEnhanceLayerWithBiasTest = np.hstack(
        [OutputOfFeatureMappingLayerTest, 0.1 * np.ones((OutputOfFeatureMappingLayerTest.shape[0], 1))])
    tempOfOutputOfEnhanceLayerTest = np.dot(InputOfEnhanceLayerWithBiasTest, weightOfEnhanceLayer)
    #  强化层输出
    OutputOfEnhanceLayerTest = tansig(tempOfOutputOfEnhanceLayerTest * parameterOfShrink)
    #  最终层输入
    InputOfOutputLayerTest = np.hstack([OutputOfFeatureMappingLayerTest, OutputOfEnhanceLayerTest])
    #  最终测试输出
    OutputOfTest = np.dot(InputOfOutputLayerTest, OutputWeight)
    time_end = time.time()  # 训练完成
    testTime = time_end - time_start
    testAcc = show_accuracy(OutputOfTest, test_y)
    print('Testing accurate is', testAcc * 100, '%')
    print('Testing time is ', testTime, 's')
    test_acc[0][0] = testAcc
    test_time[0][0] = testTime
    '''
        增量增加强化节点
    '''
    parameterOfShrinkAdd = []
    for e in list(range(L)):
        time_start = time.time()
        if N1 * N2 >= M:
            random.seed(e)
            weightOfEnhanceLayerAdd = La.orth(2 * random.randn(N2 * N1 + 1, M) - 1)
        else:
            random.seed(e)
            weightOfEnhanceLayerAdd = La.orth(2 * random.randn(N2 * N1 + 1, M).T - 1).T

        #        WeightOfEnhanceLayerAdd[e,:,:] = weightOfEnhanceLayerAdd
        #        weightOfEnhanceLayerAdd = weightOfEnhanceLayer[:,N3+e*M:N3+(e+1)*M]
        tempOfOutputOfEnhanceLayerAdd = np.dot(InputOfEnhanceLayerWithBias, weightOfEnhanceLayerAdd)
        parameterOfShrinkAdd.append(s / np.max(tempOfOutputOfEnhanceLayerAdd))
        OutputOfEnhanceLayerAdd = tansig(tempOfOutputOfEnhanceLayerAdd * parameterOfShrinkAdd[e])
        tempOfLastLayerInput = np.hstack([InputOfOutputLayer, OutputOfEnhanceLayerAdd])

        D = pinvOfInput.dot(OutputOfEnhanceLayerAdd)
        C = OutputOfEnhanceLayerAdd - InputOfOutputLayer.dot(D)
        if C.all() == 0:
            w = D.shape[1]
            B = np.mat(np.eye(w) - np.dot(D.T, D)).I.dot(np.dot(D.T, pinvOfInput))
        else:
            B = pinv(C, c)
        pinvOfInput = np.vstack([(pinvOfInput - D.dot(B)), B])
        OutputWeightEnd = pinvOfInput.dot(train_y)
        InputOfOutputLayer = tempOfLastLayerInput
        Training_time = time.time() - time_start
        train_time[0][e + 1] = Training_time
        OutputOfTrain1 = InputOfOutputLayer.dot(OutputWeightEnd)
        TrainingAccuracy = show_accuracy(OutputOfTrain1, train_y)
        train_acc[0][e + 1] = TrainingAccuracy
        print('Incremental Training Accuracy is :', TrainingAccuracy * 100, ' %')

        # 增量增加节点的 测试过程
        time_start = time.time()
        OutputOfEnhanceLayerAddTest = tansig(
            InputOfEnhanceLayerWithBiasTest.dot(weightOfEnhanceLayerAdd) * parameterOfShrinkAdd[e])
        InputOfOutputLayerTest = np.hstack([InputOfOutputLayerTest, OutputOfEnhanceLayerAddTest])

        OutputOfTest1 = InputOfOutputLayerTest.dot(OutputWeightEnd)
        TestingAcc = show_accuracy(OutputOfTest1, test_y)

        Test_time = time.time() - time_start
        test_time[0][e + 1] = Test_time
        test_acc[0][e + 1] = TestingAcc
        print('Incremental Testing Accuracy is : ', TestingAcc * 100, ' %')

    return test_acc, test_time, train_acc, train_time


def OutImg(InImg):
    ymax = 255
    ymin = 0
    xmax = np.max(InImg)
    xmin = np.min(InImg)
    OutImg = (ymax - ymin) * (InImg - xmin) / (xmax - xmin) + ymin
    return np.round(OutImg)


if __name__ == '__main__':
    # 加载数据

    dataFile = 'D:/DataSet/MNIST/mnist.mat'
    mnist = scio.loadmat(dataFile)
    traindata = np.double(mnist['train_x'] / 255)
    trainlabel = np.double(mnist['train_y'])
    testdata = np.double(mnist['test_x'] / 255)
    testlabel = np.double(mnist['test_y'])

    C = 2 ** (-30)
    s = 0.7
    N11 = 15  # Feature nodes per window
    N2 = 10  # Number of windows of feature nodes
    N33 = 5000  # Number of enhancement nodes
    epochs = 10  # Number of epochs
    M3 = 50  # Number of enhancing nodes

    # N11 = 10  # Feature nodes per window
    # N2 = 10  # Number of windows of feature nodes
    # N33 = 500  # Number of enhancement nodes
    # epochs = 10  # Number of epochs
    # M3 = 50  # Number of enhancing nodes

    train_err = np.zeros((epochs, 1))
    test_err = np.zeros((epochs, 1))
    train_time = np.zeros((epochs, 1))
    test_time = np.zeros((epochs, 1))
    N1 = N11
    N3 = N33

    for j in range(epochs):
        TrainingAccuracy, TestingAccuracy, Training_time, Testing_time = bls_train(traindata, trainlabel, testdata,
                                                                                   testlabel, s, C, N1, N2, N3)
        train_err[j] = TrainingAccuracy * 100
        test_err[j] = TestingAccuracy * 100
        train_time[j] = Training_time
        test_time[j] = Testing_time
