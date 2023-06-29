#!bin/sh
import mxnet as mx
from mxnet import gluon, init, np, npx,image
from mxnet import autograd
from mxnet.gluon import nn
from read_data import *
import sys
sys.path.append("/home/studio-lab-user/sagemaker/mxprob/")
from hamiltonian.inference.sgd import sgd
from hamiltonian.models.linear import pretrained_model,pretrained_model_aleatoric,pretrained_model_beta


learning_rate=0.01
devices=mx.gpu(0)


normalize=gluon.data.vision.transforms.Normalize()
train_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.RandomResizedCrop(224),
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    gluon.data.vision.transforms.ToTensor(),
    normalize])
test_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.Resize(256),
    gluon.data.vision.transforms.CenterCrop(224),
    gluon.data.vision.transforms.ToTensor(),
    normalize])


num_epochs = 20
train_df,test_df=read_data()
batch_size=32
train_data_loader=generate_dataset(train_df,batch_size,True,transform=train_augs)
model=pretrained_model('vgg16',{'alpha':10.0,'scale':10.0},in_units=(3,224,224),out_units=4,ctx=devices)
model2=pretrained_model('mobilenetv2_1.0',{'alpha':10.0,'scale':10.0},in_units=(3,224,224),out_units=4,ctx=devices)
model3=pretrained_model('resnet18_v2',{'alpha':10.0,'scale':10.0},in_units=(3,224,224),out_units=4,ctx=devices)

inference=sgd(model,step_size=0.001,ctx=devices)
par,loss=inference.fit(epochs=num_epochs,batch_size=batch_size,
                           data_loader=train_data_loader,
			   chain_name='vgg16_beta.h5',
                           verbose=True,metric='rmse')

inference=sgd(model2,step_size=0.001,ctx=devices)
par,loss=inference.fit(epochs=num_epochs,batch_size=batch_size,
                           data_loader=train_data_loader,
               chain_name='mobilenet_beta.h5',
                           verbose=True,metric='rmse')

inference=sgd(model3,step_size=0.001,ctx=devices)
par,loss=inference.fit(epochs=num_epochs,batch_size=batch_size,
                           data_loader=train_data_loader,
               chain_name='resnet_beta.h5',
                           verbose=True,metric='rmse')