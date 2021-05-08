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
from future_image_similarity.action_example import ImageFolder
from scipy.misc import imread
from torch.utils.data import DataLoader, Dataset

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--log_dir', default='logs', help='base directory to save logs')
parser.add_argument('--model_dir', default='', help='base directory to save trained models')
parser.add_argument('--name', default='', help='identifier for directory')
parser.add_argument('--data_root', default='data', help='root directory for data')
parser.add_argument('--niter', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--epoch_size', type=int, default=600, help='epoch size')
parser.add_argument('--image_width', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--channels', default=3, type=int)
parser.add_argument('--dataset', default='lab_value', help='critic training data: lab_value or gaz_value')
parser.add_argument('--n_past', type=int, default=3, help='number of frames to condition on')
parser.add_argument('--n_future', type=int, default=1, help='number of frames to predict during training')
parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
parser.add_argument('--prior_rnn_layers', type=int, default=1, help='number of layers')
parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
parser.add_argument('--z_dim', type=int, default=64, help='dimensionality of z_t')
parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')
parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
parser.add_argument('--data_threads', type=int, default=5, help='number of data loading threads')
parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')
parser.add_argument('--gpu', default='0', help='gpu number')
parser.add_argument('--num_cand', default=16, type=int, help='number of candidate poses for value function')

opt = parser.parse_args()

# DTW: dynamic time warping
x = np.array([1, 2, 3, 3, 7])
y = np.array([1, 2, 2, 2, 2, 2, 2, 4])
distance, path = fastdtw(x, y, dist=euclidean)

# Implement custom Dataset for each trajectory in /lab/test
class LabDTWTest(Dataset):
    """Data Handler that loads turtlebot data with value."""
    
    def __init__(self):
        self.root_dir = "future_image_similarity/data/lab"
        self.data_dir = os.path.join(self.root_dir, 'test')
        self.dirs = []
        for d1 in os.listdir(self.data_dir):
            for d2 in os.listdir('%s/%s' % (self.data_dir, d1)):
                if len(os.listdir(os.path.join(self.data_dir, d1, d2))) > 30:
                    self.dirs.append('%s/%s/%s' % (self.data_dir, d1, d2))
    
    def __len__(self):
        return len(self.dirs)
    
    def __getitem__(self, index):
        """
        :return:
        image sequence of an expert trajectory in one directory
        """
        d = self.dirs[index]

        num_files = len(os.listdir(d)) - 1
        image_seq = []
        for i in range(num_files):
            fname = os.path.join(d, 'frame'+str(i).zfill(6)+'.jpg')
            im = imread(fname).reshape(1, 64, 64, 3)
            image_seq.append(im/255.)
        image_seq = np.concatenate(image_seq, axis=0)

        return image_seq

def main():
    test_dataset = LabDTWTest()
    test_loader = DataLoader(test_dataset,
                             num_workers=opt.data_threads,
                             batch_size=1,
                             shuffle=False)
    for sequence in test_loader:
        # predict trajectory the same length as expert trajectory
        # TODO predict trajectory until reaching target
        start = sequence[0]
        predicted = MBC_predict(start, len(sequence))
        pass

def MBC_predict(start, length):
    diff_pose_rand = []  # TODO

    num_cand = 60

    predictor_pth = os.getcwd() + '/logs/lab_pose/model_predictor_pretrained/model.pth'
    critic_pth = os.getcwd() + '/logs/lab_value/model_critic_pretrained/model_critic.pth'
    image = os.getcwd() + '/dataset/office_real/airfil/run1'

    data_transforms = transforms.Compose([transforms.Scale((64,64)), transforms.ToTensor()])
    dataset = ImageFolder(image, data_transforms)

    curr_img = Variable(start.cuda(), volatile=True)

    saved_model = torch.load(predictor_pth)
    prior = saved_model['prior'].cuda()
    encoder = saved_model['encoder'].cuda()
    decoder = saved_model['decoder'].cuda()
    pose = saved_model['pose_network'].cuda()
    conv = saved_model['conv_network'].cuda()
    saved_model = torch.load(critic_pth)
    critic = saved_model['value_network'].cuda()

    for _ in range(length):
        # predict action, apply action, get next frame
        h_conv = encoder(curr_img)
        h_conv, skip = h_conv
        h = h_conv.view(-1, 4*opt.g_dim)
        z_t, _, _ = prior(h)
        z_t = z_t.view(-1, int(opt.z_dim/4), 2, 2)

        z_d_exp = pose(Variable(diff_pose_rand[0].cuda())).detach()
        h_pred_exp = conv(torch.cat([h_conv, z_t, z_d_exp], 1)).detach()
        x_pred_exp = decoder([h_pred_exp, skip]).detach()

        for j in range(num_cand):
            z_d_rand = pose(Variable(diff_pose_rand[j].cuda()))
            h_pred_rand = conv(torch.cat([h_conv, z_t, z_d_rand], 1))
            x_pred_rand = decoder([h_pred_rand, skip])
            x_value_rand = critic(x_pred_rand)
            best_pose_idx = np.argmax(-x_value_rand.data.cpu().numpy())
        
            reward_label = []
            for batch_idx in range(opt.batch_size):
                reward_label.append(loss_criterion(x_pred_rand[batch_idx], x_pred_exp[batch_idx].detach()).data)
        
            reward_label = torch.stack(reward_label)
            reward_label = Variable(reward_label, requires_grad=False)
        
            reward_loss += loss_criterion(x_value_rand, reward_label)
        
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



if __name__ == "__main__":
    main()