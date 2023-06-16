import mxnet as mx
from mxnet import gluon, init, np, npx, image
from mxnet import autograd
from mxnet.gluon import nn, loss, Trainer
from read_data import *


class Accumulator:

    """For accumulating sums over `n` variables."""

    def __init__(self, n):
        """Defined in :numref:`sec_utils`"""

        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for (a, b) in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


learning_rate = 0.01
devices = mx.gpu() 

# VGG16
pretrained_net_vgg16 = gluon.model_zoo.vision.vgg16(pretrained=True, ctx=devices)
net_vgg16 = gluon.nn.HybridSequential()  # Inicialización de API secuencial
net_vgg16.add(pretrained_net_vgg16.features)
net_vgg16.add(gluon.nn.Dense(128, in_units=4096, activation='relu'))  # Capa de salida
net_vgg16.add(gluon.nn.Dense(64, activation='relu'))  # Capa de salida
net_vgg16.add(gluon.nn.Dense(32, activation='relu'))  # Capa de salida
net_vgg16.add(gluon.nn.Dense(4))
net_vgg16[1:].initialize(init.Xavier(), ctx=devices)
net_vgg16.hybridize()
trainer_vgg16 = gluon.Trainer(net_vgg16.collect_params(), 'sgd', {'learning_rate': learning_rate, 'wd': 0.001})
train_rmse_vgg16 = gluon.metric.RMSE()
test_rmse_vgg16 = gluon.metric.RMSE()

# MobileNet
pretrained_net_mobilenet = gluon.model_zoo.vision.mobilenet_v2_1_0(pretrained=True, ctx=devices)
net_mobilenet = gluon.nn.HybridSequential()  # Inicialización de API secuencial
net_mobilenet.add(pretrained_net_mobilenet.features)
net_mobilenet.add(gluon.nn.Dense(128, in_units=1280, activation='relu'))  # Capa de salida
net_mobilenet.add(gluon.nn.Dense(64, activation='relu'))  # Capa de salida
net_mobilenet.add(gluon.nn.Dense(32, activation='relu'))  # Capa de salida
net_mobilenet.add(gluon.nn.Dense(4))
net_mobilenet[1:].initialize(init.Xavier(), ctx=devices)
net_mobilenet.hybridize()
trainer_mobilenet = gluon.Trainer(net_mobilenet.collect_params(), 'sgd', {'learning_rate': learning_rate, 'wd': 0.001})
train_rmse_mobilenet = gluon.metric.RMSE()
test_rmse_mobilenet = gluon.metric.RMSE()

# Resnet
pretrained_net_resnet = gluon.model_zoo.vision.resnet18_v2(pretrained=True, ctx=devices)
net_resnet = gluon.nn.HybridSequential()
net_resnet.add(pretrained_net_resnet.features)
net_resnet.add(gluon.nn.Dense(128, in_units=512, activation='relu'))
net_resnet.add(gluon.nn.Dense(64, activation='relu'))
net_resnet.add(gluon.nn.Dense(32, activation='relu'))
net_resnet.add(gluon.nn.Dense(4))
net_resnet[1:].initialize(init.Xavier(), ctx=devices)
net_resnet.hybridize()
trainer_resnet = gluon.Trainer(net_resnet.collect_params(), 'sgd', {'learning_rate': learning_rate, 'wd': 0.001})
train_rmse_resnet = gluon.metric.RMSE()
test_rmse_resnet = gluon.metric.RMSE()

loss = gluon.loss.L1Loss()


num_epochs = 20
train_accum = Accumulator(2)
test_accum = Accumulator(2)

for epoch in range(num_epochs):
    # VGG16
    for (i, (X_train, y_train)) in enumerate(train_data_loader):
        X_train = X_train.as_in_context(devices)
        y_train = y_train.as_in_context(devices)
        with autograd.record():
            y_pred = net_vgg16(X_train)
            loss_vgg16 = loss(y_pred, y_train)
        loss_vgg16.backward()
        train_rmse_vgg16.update(labels=y_train, preds=y_pred)
        trainer_vgg16.step(batch_size)
        rmse_val = train_rmse_vgg16.get()
    print('epoch : {}, VGG16,  train loss: {}, train RMSE: {}'.format(epoch, loss_vgg16.asnumpy().mean(), train_rmse_vgg16.get()))
    for (X_test, y_test) in test_data_loader:
        X_test = X_test.as_in_context(devices)
        y_test = y_test.as_in_context(devices)
        y_pred = net_vgg16(X_test)
        test_rmse_vgg16.update(labels=y_test, preds=y_pred)
        l_vgg16 = loss(net_vgg16(X_test), y_test)
        rmse_val = test_rmse_vgg16.get()
    print('epoch : {}, VGG16, test loss: {}, test RMSE: {}'.format(epoch, l_vgg16.asnumpy().mean(), test_rmse_vgg16.get()))

    # MobileNet
    for (i, (X_train, y_train)) in enumerate(train_data_loader):
        X_train = X_train.as_in_context(devices)
        y_train = y_train.as_in_context(devices)
        with autograd.record():
            y_pred = net_mobilenet(X_train)
            loss_mobilenet = loss(y_pred, y_train)
        loss_mobilenet.backward()
        train_rmse_mobilenet.update(labels=y_train, preds=y_pred)
        trainer_mobilenet.step(batch_size)
        rmse_val = train_rmse_mobilenet.get()
    print('epoch : {}, MobileNet, train loss: {}, train RMSE: {}'.format(epoch, loss_mobilenet.asnumpy().mean(), train_rmse_mobilenet.get()))
    for (X_test, y_test) in test_data_loader:
        X_test = X_test.as_in_context(devices)
        y_test = y_test.as_in_context(devices)
        y_pred = net_mobilenet(X_test)
        test_rmse_mobilenet.update(labels=y_test, preds=y_pred)
        l_mobilenet = loss(net_mobilenet(X_test), y_test)
        rmse_val = test_rmse_mobilenet.get()
    print('epoch : {}, MobileNet, test loss: {}, test RMSE: {}'.format(epoch,l_mobilenet.asnumpy().mean(), test_rmse_mobilenet.get()))
    
    # ResNet
    for (i, (X_train, y_train)) in enumerate(train_data_loader):
        X_train = X_train.as_in_context(devices)
        y_train = y_train.as_in_context(devices)
        with autograd.record():
            y_pred = net_resnet(X_train)
            loss_resnet = loss(y_pred, y_train)
        loss_resnet.backward()
        train_rmse_resnet.update(labels=y_train, preds=y_pred)
        trainer_resnet.step(batch_size)
        rmse_val = train_rmse_resnet.get()
    print('epoch : {}, ResNet, train loss: {}, train RMSE: {}'.format(epoch,loss_resnet.asnumpy().mean(), train_rmse_resnet.get()))
    for (X_test, y_test) in test_data_loader:
        X_test = X_test.as_in_context(devices)
        y_test = y_test.as_in_context(devices)
        y_pred = net_resnet(X_test)
        test_rmse_resnet.update(labels=y_test, preds=y_pred)
        l_resnet = loss(net_resnet(X_test), y_test)
        rmse_val = test_rmse_resnet.get()
    print('epoch : {}, ResNet, test loss: {}, test RMSE: {}'.format(epoch,l_resnet.asnumpy().mean(), test_rmse_resnet.get()))
    #print('-------------------------------------------------------------------------------------------------------------')
