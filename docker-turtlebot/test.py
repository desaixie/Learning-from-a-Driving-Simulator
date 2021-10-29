#!/usr/bin/env python
import math
from math import pi, atan2, sqrt, cos, sin
import sys

import roslib#; roslib.load_manifest('Phoebe')
import rospy
from nav_msgs.msg import Odometry
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
from actionlib_msgs.msg import *

class TurtlebotROS():
    def __init__(self):
        self.cur_pose = None
        rospy.init_node('Turtlebot_rospy', anonymous=True)
        rospy.Subscriber('odom',Odometry,self.odometryCb)
        # rospy.init_node('move', anonymous=False)
        # rospy.spin()
        rospy.on_shutdown(self.shutdown)  # what to do if shut down (e.g. ctrl + C or failure)

        #tell the action client that we want to spin a thread by default
        self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        rospy.loginfo("wait for the action server to come up")
        #allow up to 5 seconds for the action server to come up
        self.move_base.wait_for_server(rospy.Duration(5))

    def odometryCb(self, msg):
        self.cur_pose = msg.pose.pose
        rospy.loginfo(self.cur_pose)
        rate = rospy.Rate(10)  # 10hz
        rate.sleep()
        
    def shutdown(self):
        rospy.loginfo("Turtlebot is stopped")
    
    def moveTurtlebot(self, command):
        rospy.loginfo("Start moving turtlebot")
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = 'base_link'
        goal.target_pose.header.stamp = rospy.Time.now()
        # set goal position
        goal.target_pose.pose.position.x = self.cur_pose.position.x + command[0]
        goal.target_pose.pose.position.y = self.cur_pose.position.y + command[1]
        # set goal orientation
        curr_angle = 2*atan2(self.cur_pose.orientation.z, self.cur_pose.orientation.w)
        new_angle = curr_angle + command[2]
        # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Definition
        goal.target_pose.pose.orientation.w = cos(new_angle/2)
        goal.target_pose.pose.orientation.z = sin(new_angle/2)
    
        #start moving
        self.move_base.send_goal(goal)

        # allow TurtleBot up to 60 seconds to complete task
        success = self.move_base.wait_for_result(rospy.Duration(60))

        if not success:
            self.move_base.cancel_goal()
            rospy.loginfo("The base failed to move forward 3 meters for some reason")
        else:
            # We made it!
            state = self.move_base.get_state()
            if state == GoalStatus.SUCCEEDED:
                rospy.loginfo("Hooray, the base moved 3 meters forward")

if __name__ == '__main__':
    try:
        turtlebot = TurtlebotROS()
        rospy.sleep(1)
        turtlebot.moveTurtlebot([1.0, 0.0, math.pi/6.0])
        rospy.sleep(1)
        sys.exit(0)
        
    except rospy.ROSInterruptException:
        rospy.loginfo("Exception thrown")
