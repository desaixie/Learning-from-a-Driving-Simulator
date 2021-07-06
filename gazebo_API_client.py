#!/usr/bin/env python
"""
Bridge between Gazebo simulation and IL agent with rospy.
Workflow:
1. Connects with server that is running in docker
2. Receive OK from server, then load the image /docker-turtlebot/photo.jpg as current state s_t..
3. Send turtlebot command to server
4. Receive DONE, TIMEOUT, or OK.
5. If not OK, start another episode/trajectory, send RESTART, go to 2.

Simulation DTW test:
Testing set: 9+8 trajectories for 2 world files. Run 10 policy trajectories in simulation for each of the 2 world files,
each compute DTW scores with each of all trajectories and find min. Final score for each world is the average of 10 min DTW scores.
DTW score is computed using the trajectory coordinates from odom.
"""
import socket

HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 65432        # The port used by the server

testN = 0
totalTests = 10
t = 0
#1. Connects with server that is running in docker
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    s.sendall(b'RESTART')
    print(f"start test {testN}")
    testN += 1
    while True:
        # 2. Receive OK from server, then load the image /docker-turtlebot/photo.jpg as current state s_t..
        # 4. Receive DONE, TIMEOUT, or OK.
        data = s.recv(1024)
        assert data in [b"OK", b"DONE", b"TIMEOUT"]
        if data == b"OK":
            print("loaded image as current state")
            # use current state image to get action/command
        # 5. If not OK, start another episode/trajectory, send RESTART, go to 2; if all tests are done, send TERMINATE, exit..
        elif data in [b"DONE", b"TIMEOUT"]:
            if testN == totalTests:
                s.sendall(b'TERMINATE')
                print("All tests done")
                break
            else:
                s.sendall(b'RESTART')
                print(f"start test {testN}")
                testN += 1
                t = 0
                continue

        # 3. Send turtlebot command to server
        # s.sendall(b'some command' + t.to_bytes(2, byteorder='big'))
        s.sendall(f"some command {t}".encode("ascii"))
        t += 1
    
print("client exited")
