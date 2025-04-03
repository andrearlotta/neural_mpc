#!/bin/bash

uri=11311
port=10000
sessname="nmpc_multi_train#"
PANE=(ROS_LAUNCH UNITY)

if [ -z "$1" ]; then
  n_train=0
else
  n_train=$1
fi

for (( c=0; c<=$n_train; c++ ))
do
    new_port=$((port+c))
    new_uri=$((uri+c))

    sleep 1
    tmux new-session -d -s "$sessname$c"

    for ((t=0; t<${#PANE[@]}; t++))
    do
        if [ $t -gt 0 ]; then
            tmux new-window -t $sessname$c:$t
        fi
        tmux rename-window -t $sessname$c:$t ${PANE[t]}
        tmux send-keys -t $sessname$c:$t "export ROS_MASTER_URI=http://127.0.0.1:$new_uri" C-m
    done
    #  Launch nmpc_ros baseline.launch with tcp_port
    tmux send-keys -t $sessname$c:0 "source ros/pyenv/bin/activate && source ros/devel/setup.bash && roslaunch -p $new_uri nmpc_ros baseline.launch tcp_port:=$new_port" C-m

    #  Launch Unity app with ros-ip and ros-port
    UNITY_EXEC="./unity_exec/neural_mpc_unity.x86_64"
    tmux send-keys -t $sessname$c:1 "$UNITY_EXEC --ros-ip=127.0.0.1 --ros-port=$new_port" C-m

    sleep 5
done
