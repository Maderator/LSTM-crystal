#!/bin/bash

# define a DATADIR variable: directory where the input files are taken from and where output will be copied to
# TODO: Change DATADIR to your directory
DATADIR=/storage/brno2/home/maderaja/LSTM-crystal-growth/metacentrum_experiments/02_GradientThreshold

# append a line to a file "jobs_info.txt" containing the ID of the job, the hostname of node it is run on and the path to a scratch directory
# this information helps to find a scratch directory in case the job fails and you need to remove the scratch directory manually 
echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs_info.txt

# set PATH to find MATLAB
module add matlab

# test if scratch directory is set
# if scratch directory is not set, issue error message and exit
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

# copy input file "h2o.com" to scratch directory
# if the copy operation fails, issue error message and exit
cp -r $DATADIR/*  $SCRATCHDIR || { echo >&2 "Error while copying input file(s)!"; exit 2; }

# move into scratch directory
cd $SCRATCHDIR

BAYESOPT_OBJECT_FILENAME="bayesObj_${RNNType}.mat"

BAYESOBJ_PATH="results/${RNNType}/${BAYESOPT_OBJECT_FILENAME}"

if [[ -f "$BAYESOBJ_PATH" ]]; then
    echo "BayesOpt object file $BAYESOPT_OBJECT_FILENAME already exists, resuming the optimization." >> $DATADIR/jobs_info.txt
    matlab -nosplash -nodesktop -nodisplay -r "addpath('$SCRATCHDIR'); load('$BAYESOBJ_PATH'); metacentrumGradientThreshold('$RNNType', true, true, 'BayesObject', BayesObject, 'maxSeconds', $maxSeconds, 'maxObjectiveEvaluations', $maxObjectiveEvaluations, 'GradientThresholdRange', $GradientThresholdRange, 'NumberOfEpochsRange', $NumberOfEpochsRange);"
else
    echo "BayesOpt object file $BAYESOPT_OBJECT_FILENAME does not exist, starting the optimization." >> $DATADIR/jobs_info.txt
    matlab -nosplash -nodesktop -nodisplay -r "addpath('$SCRATCHDIR'); metacentrumGradientThreshold('$RNNType', false, true, 'maxSeconds', $maxSeconds, 'maxObjectiveEvaluations', $maxObjectiveEvaluations, 'GradientThresholdRange', $GradientThresholdRange, 'NumberOfEpochsRange', $NumberOfEpochsRange);"
fi



# move the output to user's DATADIR or exit in case of failure
cp -r results/$RNNType/ $DATADIR/results/ || { echo >&2 "Result file(s) copying failed (with a code $?) !!"; exit 4; }

# clean the SCRATCH directory
clean_scratch



# or in a different way:
#matlab -nosplash -nodesktop -nodisplay < lstmBayesOptimization.m > output.txt