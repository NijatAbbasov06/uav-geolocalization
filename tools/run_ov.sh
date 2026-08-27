#!/usr/bin/env bash
# Run OpenVINS on a MUN-FRL sequence and capture ALL console output to a file.
#
# Usage:  ./run_ov.sh [dataset_dir]
#   default dataset_dir = ~/workspace/munfrl/bell412_dataset3
#
# Writes: ~/workspace/munfrl/logs/<dataset>_<timestamp>.log
set -u

DS=${1:-$HOME/workspace/munfrl/bell412_dataset3}
NAME=$(basename "$DS")
LOGDIR=$HOME/workspace/munfrl/logs
mkdir -p "$LOGDIR"
LOG="$LOGDIR/${NAME}_$(date +%Y%m%d-%H%M%S).log"

# ROS/colcon setup scripts reference undefined vars (COLCON_TRACE, AMENT_TRACE...),
# so 'set -u' has to be off while they are sourced or they abort half-way.
set +u
source /opt/ros/jazzy/setup.bash
source "$HOME/workspace/catkin_ws_ov/install/setup.bash"
set -u

GT="$DS/ppk_data/${NAME}_gt.csv"
BAG="$DS/${NAME}_ros2_full"
EST="$LOGDIR/ov_estimate.txt"
STD="$LOGDIR/ov_estimate_std.txt"

{
  echo "=== run_ov.sh ==="
  echo "dataset : $DS"
  echo "bag     : $BAG"
  echo "gt      : $GT"
  echo "est out : $EST"
  echo "started : $(date -Is)"
  echo "================="
} | tee "$LOG"

# stdbuf -oL forces line buffering so the log is complete even if we ctrl-c
stdbuf -oL -eL ros2 launch ov_msckf subscribe.launch.py \
    config:=munfrl use_stereo:=false max_cameras:=1 \
    save_total_state:=true \
    filepath_est:="$EST" filepath_std:="$STD" \
    path_gt:="$GT" >> "$LOG" 2>&1 &
LAUNCH_PID=$!

sleep 5   # let the node come up and print its parameter block

stdbuf -oL ros2 bag play "$BAG" >> "$LOG" 2>&1

sleep 3
kill -INT $LAUNCH_PID 2>/dev/null
wait $LAUNCH_PID 2>/dev/null

echo
echo "done -> $LOG"
echo "lines: $(wc -l < "$LOG")"
