import os
import io
import numpy as np
import torch

import copy
from PIL import Image
# from scipy.misc import imread
from cv2 import imread
from math import pi, atan2, sqrt, cos, sin
from numpy import sign

class Gazebo(object):
    
    """Data Handler that loads turtlebot data with value."""

    def __init__(self, data_root, train=True, seq_len=2, image_size=64, skip_factor=10, N=20):
        self.root_dir = data_root
        if train:
            self.data_dir = os.path.join(self.root_dir, 'train')
            self.ordered = False
        else:
            self.data_dir = os.path.join(self.root_dir, 'test')
            self.ordered = True 
        self.dirs = []
        for d1 in os.listdir(self.data_dir):
            for d2 in os.listdir('%s/%s' % (self.data_dir, d1)):
                self.dirs.append('%s/%s/%s' % (self.data_dir, d1, d2))
        self.seq_len = seq_len
        self.image_size = image_size 
        self.seed_is_set = False
        self.d = 0
        self.skip = skip_factor
        self.N = N  # 16 in train_value opt.num_cand
        # print(f"seq len:   {self.seq_len}")  # 6
        # print(f"skip factor:   {self.skip}")  # 10

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
          
    def __len__(self):
        return len(self.dirs)
    
    def load_odom(self, dir, skip):
        diff_pose_exp = []
        with open(os.path.join(dir, dir.split('/')[-1]+'_odom.txt'), 'r') as f:
            lines = f.read().split('\n')
            # print(f"len {len(lines)} skip {skip}")
            for i in range(0, len(lines) - skip*9 - 1, skip*9):
                px = float(lines[i+1].split(': ')[-1])
                py = float(lines[i+2].split(': ')[-1])
                oz = float(lines[i+7].split(': ')[-1])
                ow = float(lines[i+8].split(': ')[-1])

                angle = 2*atan2(oz, ow)

                px_rel = float(lines[(i+1) + skip*9].split(': ')[-1])
                py_rel = float(lines[(i+2) + skip*9].split(': ')[-1])
                oz_rel = float(lines[(i+7) + skip*9].split(': ')[-1])
                ow_rel = float(lines[(i+8) + skip*9].split(': ')[-1])

                angle_rel = 2*atan2(oz_rel, ow_rel)
                        
                d_px_exp = px_rel - px
                d_py_exp = py_rel - py
                d_p_exp = sqrt(d_px_exp**2 + d_py_exp**2)

                d_angle_temp = angle_rel - angle

                if abs(d_angle_temp) > pi:
                    #crossing +/-pi border
                    d_angle = -sign(d_angle_temp)*(2*pi-abs(d_angle_temp))
                else:
                    d_angle = d_angle_temp

                diff_pose_exp.append(torch.tensor([d_px_exp, d_py_exp, d_angle], dtype=torch.float))
                # for k in range(num_cand):
                #     if k == 0:
                #         dx, dy, d_phi = d_px_exp, d_py_exp, d_angle
                #     else:
                #         dx, dy, d_phi = self.svg_gen_diff_pose_value(angle, d_p_exp)
                #
                #     diff_pose.append(torch.FloatTensor([dx, dy, d_phi]))
                #     diff_pose_exp.append(torch.FloatTensor([d_px_exp, d_py_exp, d_angle]))
                #     val_pose.append(torch.FloatTensor([abs(d_angle - d_phi)]))

        return diff_pose_exp

    def __getitem__(self, index):
        '''
        Outputs:
        
        '''
    
        d = self.dirs[index]
    
        skip = self.skip
        states = []
        num_files = len(os.listdir(os.path.join(d, 'rgb')))
        for i in range(0, num_files, skip):
            fname = os.path.join(d, 'rgb', 'frame'+str(i).zfill(6)+'.jpg')
            im = imread(fname).reshape(1, 64, 64, 3)
            im = np.moveaxis(im, 3, 1)  # move C dim
            states.append(im/255.)
        next_states = copy.deepcopy(states)
        states.pop(-1)
        next_states.pop(0)
        states = np.concatenate(states, axis=0)
        next_states = np.concatenate(next_states, axis=0)
    
        actions = self.load_odom(d, skip)
        actions = torch.stack(actions)
    
        dones = torch.zeros(len(states))
        dones[-1] = 1
        return states, actions, next_states, dones


    def get_action_space(self):
        maxR = 0
        for d in self.dirs:
            diff_pose, _, _ = self.load_odom(d, 0, self.skip, num_cand=1)
            expert_pose = diff_pose[0]
            dx, dy, _ = expert_pose
            R = dx**2 + dy**2
            if R > maxR:
                maxR = R  # R is the radius in radian
        return maxR
