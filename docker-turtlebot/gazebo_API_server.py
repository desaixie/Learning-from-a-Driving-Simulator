#!/usr/bin/env python
"""
Bridge between Gazebo simulation and IL agent with rospy.
Workflow:
1. Connects with only one client
2. Save image from turtlebot's camera, respond OK
3. Receive turtlebot command from client, translate and execute the command.
4. Check if turtlebot location is within distance from target. If so, respond DONE; if exceeds maximum timestep, respond TIMEOUT; else, go to 2.
5. Receive RESTART from client, reset Gazebo simulation and t=0, go to 2; if receive TERMINATE, exit.
Since ROS Kinetic only supports Python 2.7, the following is written in Python 2.7
When TERMINATE, close and open gazebo again with the other world file.
"""
import socket
import sys
import pickle
'''
Copyright (c) 2016, Nadya Ampilogova
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

# Script for simulation
# Launch gazebo world prior to run this script

from __future__ import print_function
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class TakePhoto:
    def __init__(self):

        self.bridge = CvBridge()
        self.image_received = False

        # Connect image topic
        img_topic = "/camera/rgb/image_raw"
        self.image_sub = rospy.Subscriber(img_topic, Image, self.callback)

        # Allow up to one second to connection
        rospy.sleep(1)

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            # TODO: resize from 640*480 to 64*64
        except CvBridgeError as e:
            print(e)

        self.image_received = True
        self.image = cv_image

    def take_picture(self, img_title):
        if self.image_received:
            # Save an image
            cv2.imwrite(img_title, self.image)
            return True
        else:
            return False


if __name__ == '__main__':
    # Initialize
    rospy.init_node('take_photo', anonymous=False)
    camera = TakePhoto()

    # Take a photo
    img_title = 'photo.jpg'
    if camera.take_picture(img_title):
        rospy.loginfo("Saved image " + img_title)
    else:
        rospy.loginfo("No images received")

    # Sleep to give the last log messages time to be sent
    rospy.sleep(1)

    # HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
    HOST = '0.0.0.0'  #  All interfaces, since server in run in a docker container
    PORT = 65432        # Port to listen on (non-privileged ports are > 1023)

    try:
        # with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:  # does no work in python2
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((HOST, PORT))
        s.listen(1)  # python2 socket.listen requires parameter: number of max queued connect requests
        print("listening on HOST " + HOST + ", PORT " + str(PORT))
        conn, addr = s.accept()
        # with conn:
        print('Connected by', addr)

        data = conn.recv(1024)
        assert data == b"RESTART"
        while True:
            t = 0  # timestep in Gazebo simulation
            max_timestep = 30
            target_coord = 10 # TODO read target coordinate from ROS
            while True:
                
                # 2. Save image from turtlebot's camera, respond OK
                print("image saved")
                conn.sendall(b"OK")
                
                # 3. Receive turtlebot command from client, translate and execute the command.
                data = conn.recv(1024)
                print("received command: ", pickle.loads(data))
                # TODO start saving odometry data until command is done (if odom is not changing.
                t += 1
                
                # 4. Check if turtlebot location is within distance from target. If so, respond DONE; if exceeds maximum timestep, respond TIMEOUT; else, go to 2.
                coord = 0  # TODO: read turtlebot coordinate from ROS
                # TODO also save a odom file
                if abs(target_coord - coord) < 3:
                    conn.sendall(b"DONE")
                    break
                elif t >= max_timestep:
                    conn.sendall(b"TIMEOUT")
                    break

            # 5. Receive RESTART from client, reset Gazebo simulation and t=0, go to 2; if receive TERMINATE, exit.
            data = conn.recv(1024)
            assert data in [b"RESTART", b"TERMINATE"]
            if data == "TERMINATE":
                break
            # TODO reset gazebo simulation
            
            # if not data:
            #     break
    except:
        print("Unexpected error:", sys.exc_info()[0])
        raise
    finally:
        # Clean up the connection
        conn.close()
    print("server exited")

