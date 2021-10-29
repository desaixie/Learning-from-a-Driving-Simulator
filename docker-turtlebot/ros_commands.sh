#!/bin/bash
# launch turtlebot using world file obs1_target1
roslaunch turtlebot_gazebo turtlebot_world.launch world_file:=/home/root/docker-turtlebot/world/obs1_target1.world

# launch turtlebot using world file obs2_target1
#roslaunch turtlebot_gazebo turtlebot_world.launch world_file:=/home/root/docker-turtlebot/world/obs2_target1.world

# launch rviz. Do this after turtlebot gazebo simulation is running
#roslaunch turtlebot_rviz_launchers view_robot.launch

# Launch turtlebot keyboard control 
#roslaunch turtlebot_teleop keyboard_teleop.launch
