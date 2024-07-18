#!/bin/bash

seconds=$1

hours=$(($seconds / 3600))
minutes=$((($seconds % 3600) / 60 ))
seconds=$(($seconds % 60))

printf "%02d:%02d:%02d\n" $hours $minutes $seconds

