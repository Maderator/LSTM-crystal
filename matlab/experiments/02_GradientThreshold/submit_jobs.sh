#!/bin/bash

# Default RNN types to run
RNN_TYPES=("LSTM" "GRU" "peepholeLSTM" "residualLSTM")

MAX_SECONDS=$((24*60*60))

MAX_OBJECTIVE_EVALUATIONS=105

GRADIENT_THRESHOLD_RANGE="[0.001 1000]"
NUMBER_OF_EPOCHS_RANGE="[10 500]"

if [ "$#" -ge 1 ]; then
    RNN_TYPES=("$1")
fi

if [ "$#" -ge 2 ]; then
    MAX_OBJECTIVE_EVALUATIONS="$2"
fi

if [ "$#" -ge 3 ]; then
    GRADIENT_THRESHOLD_RANGE="$3"
fi

if [ "$#" -ge 4 ]; then
    NUMBER_OF_EPOCHS_RANGE="$4"
fi

hours=$((($MAX_SECONDS / 3600) + 6))
minutes=$((($MAX_SECONDS % 3600) / 60 ))
seconds=$(($MAX_SECONDS % 60))
walltime=$(printf "%02d:%02d:%02d\n" $hours $minutes $seconds)

for rnn_type in "${RNN_TYPES[@]}"; do
    qsub -l select=1:ncpus=16:mem=80gb:scratch_local=8gb -l walltime=$walltime -l matlab=1 \
        -l matlab_Statistics_Toolbox=1 -l matlab_Distrib_Computing_Toolbox=1 -l matlab_Neural_Network_Toolbox=1 \
        -m abe -M janmadera97@gmail.com \
        -N "experiment02_gradientThreshold_$rnn_type" \
        -o "exp01_job_output_$rnn_type.log" \
        -e "exp01_job_error_$rnn_type.log" \
        -v RNNType="$rnn_type",maxSeconds="$MAX_SECONDS",maxObjectiveEvaluations="$MAX_OBJECTIVE_EVALUATIONS",GradientThresholdRange="$GRADIENT_THRESHOLD_RANGE",NumberOfEpochsRange="$NUMBER_OF_EPOCHS_RANGE" \
        matlab_training.sh
done
#qsub -I -l select=1:ncpus=16:mem=80gb:scratch_local=8gb -l walltime=32:30:00 -l matlab=1 -l matlab_Statistics_Toolbox=1 -l matlab_Distrib_Computing_Toolbox=1 -l matlab_Neural_Network_Toolbox=1 -m abe -M janmadera97@gmail.com -N "experiment02_gradientThreshold_$rnn_type" -o "exp01_job_output_$rnn_type.log" -e "exp01_job_error_$rnn_type.log" -v RNNType="$rnn_type",maxSeconds="$MAX_SECONDS",maxObjectiveEvaluations="$MAX_OBJECTIVE_EVALUATIONS" matlab_training.sh
