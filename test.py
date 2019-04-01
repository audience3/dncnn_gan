
import torch
import numpy as np
import argparse

import model
import torch.nn as nn
from matplotlib import pyplot as plt
import torch.utils.data
import torchvision.utils as vutils
import torch.optim as optim
import math
from torchvision import transforms, datasets,models
import pytorch_ssim



#make a parser
parser=argparse.ArgumentParser()
parser.add_argument('--gpu',default=-1,type=int)
parser.add_argument('--start',default=40,type=int)   #the stratpoint means the currently epoch,and load the last epoch data
parser.add_argument('--resume',default=False,type=bool)

params=parser.parse_args()




######################################################################
manual_seed = 999
global_step=0

# data_dir = 'data/train_set'
data_dir='/Users/audience/Desktop/cv/own_project/perceptual/result/reveals'
origin_dir='/Users/audience/Desktop/cv/own_project/perceptual/result/secrets'
model_dir= 'model'
# hyperparas definition
batch_size=4
img_size = 256
learning_rate=0.0001
weight_decay=1e-5
epochs=80
gpu=params.gpu
gpu_available =  True if gpu>=0 else False
resume=params.resume


device=torch.device("cuda:%d"%(gpu) if gpu_available else "cpu")


data_transform=transforms.Compose([
    transforms.Resize(img_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor()
    ])


dataset=datasets.ImageFolder(data_dir,data_transform)

dataloader=torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=1)

dataset_origin=datasets.ImageFolder(origin_dir,data_transform)

dataloader_origin=torch.utils.data.DataLoader(dataset_origin,batch_size=batch_size,shuffle=False,num_workers=1)




def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant(m.bias.data, 0.0)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # m.weight.data.normal_(0.0, 0.02)
        nn.init.xavier_uniform_(m.weight)
    elif classname.find('BatchNorm') != -1:
        # nn.init.xavier_uniform(m.weight)
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

#the denoising network
##############################################

dncnn=model.DnCNN()
dncnn=dncnn.to(device)
dncnn.apply(weights_init_kaiming)

#inception v3
sigmoid=nn.Sigmoid()
discriminator=models.Inception3(num_classes=1,aux_logits=False)
discriminator=discriminator.to(device)
# discriminator.apply(weights_init)



ssim=pytorch_ssim.SSIM()
bce_loss=nn.BCELoss()
mse_loss=nn.MSELoss()

optimizerG=optim.Adam(dncnn.parameters(),lr=learning_rate)
optimizerD=optim.Adam(discriminator.parameters(),lr=3*learning_rate)




network_loss=[]
ssim_=[]
mse_=[]




#loading checkpoint
if resume:
    print('loading params')
    start_epoch = params.start
    path = model_dir + '/' + str(start_epoch-1) + '.pth.tar'    #load the last epoch params
    checkpoint= torch.load(path)
    dncnn.load_state_dict(checkpoint['dncnn_state_dict'])
    discriminator.load_state_dict(checkpoint['inception_state_dict'])


    optimizerD.load_state_dict(checkpoint['incep_optim'])
    optimizerG.load_state_dict(checkpoint['dncnn_optim'])
    epoch=checkpoint['epoch']
    # cover_ssmi=checkpoint['cover_ssmi']
    # secret_ssmi=checkpoint['secret_ssmi']
    network_loss=checkpoint['net_loss']
    mse_=checkpoint['mse']
    ssim_=checkpoint['ssim']
    dncnn.train()
    discriminator.train()


    # assert start_epoch==start_epoch

else:
    start_epoch=1







for epoch in range(start_epoch,epochs):


    if epoch>=20 and epoch %3==0:
        optimizerD.param_groups[0]['lr'] *= 0.9
        optimizerG.param_groups[0]['lr'] *= 0.9

    for i,data in enumerate(zip(dataloader,dataloader_origin)):
        images = data[0][0]
        origin = data[1][0]

        if len(images) != batch_size: break
        images = 0 + 0.299 * images[:, 0, :, :] + 0.587 * images[:, 1, :, :] + 0.114 * images[:, 2, :, :]
        images = images.view(-1, 1, 256, 256)
        origin = 0 + 0.299 * origin[:, 0, :, :] + 0.587 * origin[:, 1, :, :] + 0.114 * origin[:, 2, :, :]
        origin = origin.view(-1, 1, 256, 256)


        dncnn.zero_grad()


        denoised=dncnn(images)
        ssim0=ssim(denoised,origin).item()
        ssim1=ssim(images,origin).item()

#train G
        cls_fake=sigmoid(discriminator(denoised))

        true_label=torch.ones(cls_fake.size(),device=device)
        errG=bce_loss(cls_fake,true_label)
        mse=mse_loss(denoised,origin)
        loss=mse*100+errG
        loss.backward()

        optimizerG.step()


#train D

        if i%3==0:
            discriminator.zero_grad()
            fake=denoised.detach()


            cls_fake=sigmoid(discriminator(fake))
            fake_label = torch.zeros(cls_fake.size(), device=device)
            error_fake=bce_loss(cls_fake,fake_label)

            cls_real=sigmoid(discriminator(origin))
            error_real=bce_loss(cls_real,true_label)

            errors=error_fake+error_real
            errors.backward()
            optimizerD.step()



        if i%1==0:
            print('epoch:%d  batch:%d|| ssim: %.4f ~ %.4f || loss: %.4f '%(epoch,i+1,ssim0,ssim1,loss.item()))
            mse_.append(mse.item())
            ssim_.append(ssim0)
            network_loss.append(loss.item())



    if  epoch%2==0:
        path=model_dir+'/'+str(epoch)+'.pth.tar'
        torch.save({
            'epoch':epoch,
            'dncnn_state_dict':dncnn.state_dict(),
            'inception_state_dict':discriminator.state_dict(),

            'dncnn_optim':optimizerG.state_dict(),
            'incep_optim':optimizerD.state_dict(),

            'ssim':ssim_,
            'mse':mse_,
            'net_loss':network_loss

        },path)




