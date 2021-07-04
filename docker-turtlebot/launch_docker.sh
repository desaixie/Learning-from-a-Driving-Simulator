#!/bin/bash
xhost +local:root

container_name='docker-turtlebot2dev'
curdir="$(pwd)"
#workdir="$(dirname "$curdir")"  # /path/to/.../Learning-from-a-Driving-Simulator

# create container if does not exist
[[ $(docker ps -f "name=$container_name" --format '{{.Names}}') == $container_name ]] ||
# -d runs this container in the background
# simply change nvidia-docker to docker to run the container in plain Docker without Nvidia GPU support
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
-v $curdir:/home/$USER/docker-turtlebot2 \
-e LOCAL_USER_ID=`id -u $USER` \
-e LOCAL_GROUP_ID=`id -g $USER` \
-e LOCAL_GROUP_NAME=`id -gn $USER` \
--name "$container_name" \
 docker-turtlebot:test

# start the container if stopped
[[ $(docker ps -f "name=$container_name" -f status=exited --format '{{.Names}}') == $container_name ]] &&  # and instead of or here!!
docker start $container_name

# to open a new terminal on this container:
docker exec -it "$container_name" bash

xhost -local:root
