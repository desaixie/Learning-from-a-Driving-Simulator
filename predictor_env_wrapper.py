"""
Wrap the predictor as a gym env
"""
import sys
sys.path.insert(0, './future_image_similarity')
import argparse
import random
import os

import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import future_image_similarity.utils as utils

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# ---------------- Settings for loading gaz_value dataset  ----------------
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
parser.add_argument('--dataset', default='gaz_value', help='critic training data: lab_value or gaz_value')
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

print("Initializing DreamGazeboEnv...")

# load model
if 'lab' in opt.dataset:
    opt.model_dir = 'logs/lab_pose/model_predictor_pretrained'
elif 'gaz' in opt.dataset:
    opt.model_dir = 'logs/gaz_pose/model_predictor_pretrained'  # loading model saved with new pytorch version
saved_model = torch.load('%s/model.pth' % opt.model_dir)  # https://github.com/pytorch/pytorch/issues/3678
model_dir = opt.model_dir
niter = opt.niter
epoch_size = opt.epoch_size
log_dir = opt.log_dir
num_cand = opt.num_cand
lr = opt.lr
dataset = opt.dataset
# applying newly specified options
opt = saved_model['opt']
opt.model_dir = model_dir
opt.niter = niter
opt.epoch_size = epoch_size
opt.log_dir = log_dir
opt.num_cand = num_cand
opt.lr = lr
opt.dataset = dataset

name = 'model_critic'

opt.log_dir = '%s/%s/%s' % (opt.log_dir, opt.dataset, name)
os.makedirs(opt.log_dir, exist_ok=True)

print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
dtype = torch.cuda.FloatTensor

# ---------------- load the models  ----------------
opt.n_past = 1
opt.n_future = 1  # SQIL training uses only the state and next state
opt.batch_size = 1
opt.data_root = "future_image_similarity/data/sim/targets/target1"
print(opt)

prior = saved_model['prior']
encoder = saved_model['encoder']
decoder = saved_model['decoder']
pose_network = saved_model['pose_network']
conv_network = saved_model['conv_network']
# from models.model_value import ModelValue
# value_network = ModelValue()
# value_network.apply(utils.init_weights)

# ---------------- optimizers ----------------
# opt.optimizer = optim.Adam
# value_network_optimizer = opt.optimizer(value_network.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# --------- loss functions -------------------
loss_criterion = nn.SmoothL1Loss()

# --------- transfer to gpu ------------------
prior.to(device)
encoder.to(device)
decoder.to(device)
pose_network.to(device)
conv_network.to(device)
# value_network.to(device)
loss_criterion.to(device)
prior.eval()  # eval avoid error on no tracked stat in BatchNorm2d, a feater added in later versions of pytorch
encoder.eval()
decoder.eval()
pose_network.eval()
conv_network.eval()

# --------- load a dataset -------------------
train_data, test_data = utils.load_dataset(opt)

train_loader = DataLoader(train_data,
                          num_workers=opt.data_threads,
                          batch_size=opt.batch_size,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)
test_loader = DataLoader(test_data,
                         num_workers=opt.data_threads,
                         batch_size=opt.batch_size,
                         shuffle=True,
                         drop_last=True,
                         pin_memory=True)
#
# def get_training_batch():
#     while True:
#         for sequence in train_loader:
#             batch = utils.normalize_data(opt, dtype, sequence[0])
#             yield batch, sequence[1], sequence[2]
# training_batch_generator = get_training_batch()
#
# def get_testing_batch():
#     while True:
#         for sequence in test_loader:
#             batch = utils.normalize_data(opt, dtype, sequence[0])
#             yield batch, sequence[1], sequence[2]
# testing_batch_generator = get_testing_batch()
# x, _, _ = next(training_batch_generator)  # x, list of 6 tensors of size (60, 3, 64, 64), batch_size=60
# print(type(x[0]))

class ObsSpace():
    def __init__(self, name):
        if name != "Gazebo":
            raise Exception(f"{name} is not supported")
        self.shape = np.array([64, 64, 3])
        self.high = 255. * np.ones((64, 64, 3))
        self.low = np.zeros((64, 64, 3))
    
class ActSpace():
    def __init__(self, name):
        if name != "Gazebo":
            raise Exception(f"{name} is not supported")
        self.shape = np.array([3])  # (dx, dy, theta): movement in x coordinate, movement in y coordinate, angle of direction change
        self.max_angle = 0.17453292519943295*3  # 10*(pi/180)*3, 30 deg in rad, specified in gaz_value.py
        self.max_R = 0.0012  # according to Gazebo.get_action_space()
        # no need change action space according to skip_factor=10, since predictor is trained with skip and non-scaled actions
        self.high = np.array([self.max_R, self.max_R, self.max_angle])
        self.low = np.array([-self.max_R, -self.max_R, -self.max_angle])
        
    def sample(self):
        dx = np.random.uniform(-self.max_R, self.max_R)
        dy = np.random.uniform(-self.max_R, self.max_R)
        theta = np.random.uniform(-self.max_angle, self.max_angle)
        return np.array([dx, dy, theta])
    
class DreamGazeboEnv():
    def __init__(self):
        self.observation_space = ObsSpace("Gazebo")
        self.action_space = ActSpace("Gazebo")
        self.state = None
        self.prior = prior
        self.encoder = encoder
        self.decoder = decoder
        self.pose_network = pose_network
        self.conv_network = conv_network
        
        
        self.episode_len = -1
        self.lengths = []
        for d1 in train_data.dirs:
            self.lengths.append(len(os.listdir(d1 + "/rgb")))
        # provide opt of train_value to my custom files
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.opt = opt
    
    def reset(self):
        """Randomly restart the environment to initial state of a training trajectory, return state, set corresponding episode length"""
        traj_id = random.randrange(len(self.lengths))
        # predictor for Gazebo is trained with seq_len = 5+1, skip = 10.
        self.episode_len = self.lengths[traj_id] // 10  # divide by skip_factor=10
        d = train_data.dirs[traj_id]
        fname = os.path.join(d, 'rgb', 'frame'+str(0).zfill(6)+'.jpg')  # the first image in trajectory
        im = np.asarray(PIL.Image.open(fname)).reshape((1, 64, 64, 3))  # PIL.Image.open read in RGB order
        # im = imread(fname).reshape(1, 64, 64, 3)  # cv2.imread reads in BGR order
        self.state = torch.tensor(im / 255., dtype=torch.float, device=device).permute(0, 3, 1, 2)  # make state ready to passed into encoder
        return self.state
    
    def step(self, action):
        """Step in the state-transition model given action, returns next_state, reward, done, info"""
        action = torch.tensor(action, dtype=torch.float, device=device)
        # initialize the hidden state.
        with torch.no_grad():
            self.prior.hidden = prior.init_hidden(batch_size=1)  # https://stackoverflow.com/a/58177979/8667103

            h_conv = self.encoder(self.state)
            h_conv, skip = h_conv
            h = h_conv.view(-1, 4*opt.g_dim)
            z_t, _, _ = self.prior(h)
            z_t = z_t.view(-1, int(opt.z_dim/4), 2, 2)

            z_d = self.pose_network(action).detach()
            h_pred = self.conv_network(torch.cat([h_conv, z_t, z_d], 1)).detach()
            x_pred = self.decoder([h_pred, skip]).detach()
            self.state = x_pred
        self.episode_len -= 1
        done = False
        if self.episode_len == 0:
            done = True

        # reward function design depend on the applied Imitation Learning algorithm
        return self.state, None, done, None
    
    def render(self):
        raise Exception("render is not supported")
# d = DreamGazeboEnv()
# print(train_data.get_action_space())  # maxR = 0.0012