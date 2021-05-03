from __future__ import print_function

import sys
import numpy as np
from numpy import sign
from math import sin, cos, pi

import os
import os.path
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import torch.utils.data as data
from torch.autograd import Variable
from torchvision import transforms

from PIL import Image
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from f

x = np.array([1, 2, 3, 3, 7])
y = np.array([1, 2, 2, 2, 2, 2, 2, 4])

distance, path = fastdtw(x, y, dist=euclidean)


candidates = 60

predictor_pth = os.getcwd() + '/logs/lab_pose/model_predictor_pretrained/model.pth'
critic_pth = os.getcwd() + '/logs/lab_value/model_critic_pretrained/model_critic.pth'
image = os.getcwd() + '/dataset/office_real/airfil/run1'

data_transforms = transforms.Compose([transforms.Scale((64,64)), transforms.ToTensor()])
dataset = ImageFolder(image, data_transforms)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=candidates, shuffle=False, num_workers=1)
dataloaders = {'test': dataloader}
for data in dataloaders['test']:
    inputs = data

curr_img = Variable(inputs[0].cuda(), volatile=True)
diff_pose = Variable(inputs[1].cuda(), volatile=True)

saved_model = torch.load(predictor_pth)
prior = saved_model['prior'].cuda()
encoder = saved_model['encoder'].cuda()
decoder = saved_model['decoder'].cuda()
pose = saved_model['pose_network'].cuda()
conv = saved_model['conv_network'].cuda()
saved_model = torch.load(critic_pth)
critic = saved_model['value_network'].cuda()

h_conv = encoder(curr_img)
h_conv, skip = h_conv
h = h_conv.view(-1, 4*128)
z_t, _, _ = prior(h)
z_t = z_t.view(-1, int(64/4), 2, 2)
z_d = pose(diff_pose)
h_pred = conv(torch.cat([h_conv, z_t, z_d], 1))
x_pred = decoder([h_pred, skip])
x_value = critic(x_pred)

best_pose_idx = np.argmax(-x_value.data.cpu().numpy())
delta_pose = inputs[1].cpu().numpy().tolist()[best_pose_idx]

print('action [dx, dy, d_angle]: ', delta_pose)



