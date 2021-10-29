"""
Wrap the gazebo simulation as a gym env
"""
import socket
import pickle

import PIL.Image
import numpy as np
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

class SimGazeboEnv():
    def __init__(self):
        self.observation_space = ObsSpace("Gazebo")
        self.action_space = ActSpace("Gazebo")
        
        HOST = '127.0.0.1'  # The server's hostname or IP address
        PORT = 65432        # The port used by the server
        #1. Connects with server that is running in docker
        # with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((HOST, PORT))
        self.t = 0
    
    def reset(self):
        self.s.sendall(b'RESTART')
        self.t = 0
        # 2. Receive OK from server, then load the image docker-turtlebot/photo.jpg as current state s_t..
        data = self.s.recv(1024)
        assert data == b"OK"
        fname = "docker-turtlebot/turtlebot/photo.jpg"
        with PIL.Image.open(fname) as im:
            img = np.asarray(im).reshape((1, 64, 64, 3))  # PIL.Image.open read in RGB order. Use with so file is closed!
        next_state = torch.tensor(img / 255., dtype=torch.float, device=device).permute(0, 3, 1, 2)  # make state ready to passed into encoder
        done = False
        return next_state, None, done, None
    
    def step(self, action):
        """returns next_state, reward, done, info"""
        # 3. Send turtlebot command to server
        command = pickle.dumps(action, protocol=2)  # python2 only supports up to ver 2
        self.s.sendall(command)
        self.t += 1
        # 4. Receive DONE, TIMEOUT, or OK.
        data = self.s.recv(1024)
        assert data in [b"OK", b"DONE", b"TIMEOUT"]
        if data == b"OK":
            fname = "docker-turtlebot/turtlebot/photo.jpg"
            with PIL.Image.open(fname) as im:
                img = np.asarray(im).reshape((1, 64, 64, 3))  # PIL.Image.open read in RGB order. Use with so file is closed!
            next_state = torch.tensor(img / 255., dtype=torch.float, device=device).permute(0, 3, 1, 2)  # make state ready to passed into encoder
            done = False
            # use current state image to get action/command
        # 5. If not OK, start another episode/trajectory, send RESTART, go to 2; if all tests are done, send TERMINATE, exit..
        else:
            next_state = None
            done = True
        
        return next_state, None, done, None
    
    def close(self):
        self.s.sendall(b'TERMINATE')
        self.s.close()
        print("All tests done")
