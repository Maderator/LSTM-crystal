function BayesObject = gradientThreshold(RNNType, continueWithLearning, options)
%baselineEpochBatchGrid grid search for the best size of minibatch and number of epochs
%   arguments:
%       RNNType:
%           Type of the RNN used in each layer of network:
%               'GRU', 'LSTM', 'peepholeLSTM', 'residualLSTM'
%       continueWithLearning:
%           If true, options.BayesObject has to be specified and the
%           algorithm will continue with optimization
%       options:
%           Optional arguments:
%               maxSeconds: maximum time in seconds for whole bayes optimization (default 24*60*60 = 1 day)
%               maxEpochs: maximum number of epochs for each training (default 100)
%               timewindowSize: size of the timewindow of the input data (default 4)
%               MiniBatchSize: size of the minibatch (number of experiments in batch), default 8
%   returns:
%       BayesObject: bayes object with the results of the optimization
arguments
    RNNType (1,1) string {mustBeMember(RNNType, {'GRU', 'LSTM', 'peepholeLSTM', 'residualLSTM'})} = 'LSTM'
    continueWithLearning (1,1) logical = false
    options.BayesObject (1,1) = false
    
    % Data options
    options.datasetPath (1,1) string = "trainingShuffled.mat"
    options.timewindowSize (1,1) double {mustBeInteger, mustBePositive} = 4
    
    % Bayes Optimization options
    options.GradientThresholdRange (1,2) double {mustBePositive} = [0.001 1000]
    options.NumberOfEpochsRange (1,2) double {mustBeInteger, mustBePositive} = [10 500]
    options.maxSeconds (1,1) double {mustBePositive} = Inf
    options.MaxObjectiveEvaluations (1,1) double {mustBeInteger, mustBePositive} = 100
    options.NumSeedPoints (1,1) double {mustBeInteger, mustBePositive} = 15
    options.doParallelBayes (1,1) logical = true
    options.SaveBayesObject (1,1) logical = true
    
    % Network options
    options.NetworkDepth (1,1) double {mustBeInteger, mustBePositive} = 1
    options.NumHiddenUnits (1,1) double {mustBeInteger, mustBePositive} = 100
    options.bidirectional (1,1) logical = false
    options.useDropout (1,1) logical = false
    options.useLayerNormalization (1,1) logical = false
    
    % Training options
    %options.NumberOfEpochs (1,1) double {mustBeInteger, mustBePositive} = 100
    options.MiniBatchSize (1,1) double {mustBeInteger, mustBePositive} = 8
    options.useCrossValidation (1,1) logical = true
    options.crossValidationFolds (1,1) double {mustBeInteger, mustBePositive} = 5
    options.saveModels (1,1) logical = false
end

disp("Continue with learning:" + continueWithLearning)
disp("GradientThresholdRange: " + options.GradientThresholdRange)
disp("NumberOfEpochsRange: " + options.NumberOfEpochsRange)

optimVars = [
    optimizableVariable('GradientThreshold', options.GradientThresholdRange, 'Transform', 'log')
    optimizableVariable('NumberOfEpochs', options.NumberOfEpochsRange, 'Type','integer')
    ];

data = setupData( ...
    datasetPath=options.datasetPath, ...
    timewindowSize=options.timewindowSize, ...
    useTrainTestSplit=~options.useCrossValidation, ...
    trainTestSplit = 0.95);

ObjFcn = makeObjectiveFunction(data, options.timewindowSize, RNNType, options);

% testObjFcn(ObjFcn)

if continueWithLearning
    if ~isa(options.BayesObject, "BayesianOptimization")
        errorStruct.message = 'Error. \n Missing BayesObject! Cannot resume BayesianOptimization. Ending experiment';
        errorStruct.identifier = 'gradientThreshold:BayesObjectNotSpecified';
        error(errorStruct)
    end
    % recreate bayesObject to prevent @UNKNOWN function
    usedTime = options.BayesObject.TotalElapsedTime;
    usedObjectiveEvaluations = options.BayesObject.NumObjectiveEvaluations;
    
    tic
    BayesObject = bayesopt(ObjFcn, optimVars, ...
        'InitialX', options.BayesObject.XTrace, ...
        'InitialObjective', options.BayesObject.ObjectiveTrace, ...
        'InitialConstraintViolations', options.BayesObject.ConstraintsTrace, ...
        'InitialErrorValues', options.BayesObject.ErrorTrace, ...
        'InitialUserData', options.BayesObject.UserDataTrace, ...
        'InitialObjectiveEvaluationTimes', options.BayesObject.ObjectiveEvaluationTimeTrace, ...
        'InitialIterationTimes', options.BayesObject.IterationTimeTrace, ...
        'PlotFcn', {@plotObjectiveModel, @plotMinObjective}, ...
        'MaxTime', usedTime + options.maxSeconds, ...
        'MaxObjectiveEvaluations', usedObjectiveEvaluations + options.MaxObjectiveEvaluations,...
        'isObjectiveDeterministic', false, ... % with same data and same hyperparameters, the result of more trainNetwork function calls can be different
        'NumSeedPoints', options.NumSeedPoints, ...
        'UseParallel', options.doParallelBayes);
    toc
    
    %BayesObject = resume(BayesObject, 'MaxObjectiveEvaluations', 32, 'VariableDescriptions', optimVars, 'PlotFcn', {@plotObjectiveModel, @plotMinObjective});
else
    tic
    BayesObject = bayesopt(ObjFcn, optimVars, ...
        'PlotFcn', {@plotObjectiveModel, @plotMinObjective}, ...
        'MaxTime', options.maxSeconds, ...
        'MaxObjectiveEvaluations',options.MaxObjectiveEvaluations,...
        'isObjectiveDeterministic', false, ... % with same data and same hyperparameters, the result of more trainNetwork function calls can be different
        'NumSeedPoints', options.NumSeedPoints, ...
        'UseParallel', options.doParallelBayes);
    toc
end

if options.SaveBayesObject
    disp("Saving BayesObject to file results/" + RNNType + "/bayesObj_" + RNNType + ".mat")
    save("results/" + RNNType + "/bayesObj_" + RNNType + ".mat", "BayesObject")
end

end