#!/bin/bash
xhost +local:root

container_name='ros-kinetic-gazebo7dev'

# create container if not existing
[[ $(docker ps -f "name=$container_name" --format '{{.Names}}') == $container_name ]] ||
# -d runs this container in the background
nvidia-docker run -it -d \
--env="DISPLAY"  \
--env="QT_X11_NO_MITSHM=1"  \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
--workdir="/home/$USER" \
--volume="/home/$USER:/home/$USER" \
--volume="/etc/group:/etc/group:ro" \
--volume="/etc/passwd:/etc/passwd:ro" \
--volume="/etc/shadow:/etc/shadow:ro" \
--volume="/etc/sudoers.d:/etc/sudoers.d:ro" \
--volume="$HOME/host_docker:/home/user/host_docker" \
-v /home/desaixie/Learning-from-a-Driving-Simulator/rosdocker:/home/$USER/desaixie \
-e LOCAL_USER_ID=`id -u $USER` \
-e LOCAL_GROUP_ID=`id -g $USER` \
-e LOCAL_GROUP_NAME=`id -gn $USER` \
--name "$container_name" \
 ros-kinetic-gazebo7:custom

# start the container if stopped
[[ $(docker ps -f name=ros-kinetic-gazebo7dev -f status=exited --format '{{.Names}}') == $container_name ]] &&  # and instead of or here!!
docker start $container_name

# to open a new terminal on this container:
docker exec -it ros-kinetic-gazebo7dev bash

xhost -local:root
