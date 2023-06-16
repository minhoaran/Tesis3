#!bin/sh
import mxnet as mx
from mxnet import gluon, init, np, npx,image
from mxnet import autograd
from mxnet.gluon import nn

import pandas as pd

        
def read_data(csv_fname='annotations.csv'):
    csv_data = pd.read_csv(csv_fname,header=None)
    csv_data.columns=['images','labels','bbox']
    train=csv_data.sample(frac=0.8)
    test=csv_data.drop(train.index)
    return train,test

def generate_dataset(data_frame,batch_size=32,is_train=True,transform=None):

    train_list=train_df[['bbox','images']].values.tolist()

    train_list_dec=[[eval(bbox),img] for bbox,img in train_list]
    
    train_ds=mx.gluon.data.vision.datasets.ImageListDataset(root='./data', imglist=train_list_dec) 
    
    train_data_loader = mx.gluon.data.DataLoader(train_ds.transform_first(transform), batch_size=batch_size, shuffle=is_train, num_workers=4)

    return train_data_loader


#############################################################################################################


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



batch_size=32


train_df,test_df=read_data()
train_data_loader=generate_dataset(train_df,batch_size,True,transform=train_augs)
test_data_loader=generate_dataset(test_df,batch_size,False,transform=test_augs)

