function BayesObject = metacentrumGradientThreshold(RNNType, continueWithLearning, useMetaParPool, options)
%metacentrumGradientThreshold bayes optimization of LSTM network hyperparameter gradientThreshold using Metacentrum cluster
%   arguments:
%       RNNType:
%           Type of the RNN used in each layer of network:
%               'GRU', 'LSTM', 'peepholeLSTM', 'residualLSTM'
%       useMetaParPool:
%           if true, the script uses MetaParPool function provided by
%           Metacentrum
%       options:
%           Optional arguments:
%               maxSeconds: maximum time in seconds for whole bayes optimization (default 24*60*60 = 1 day)
%               maxEpochs: maximum number of epochs for each training (default 100)
%               timewindowSize: size of the timewindow of the input data (default 4)
%               miniBatchSize: size of the minibatch (number of experiments in batch), default 8
%   returns:
%       BayesObject: bayes object with the results of the optimization
arguments
    RNNType (1,1) string {mustBeMember(RNNType, {'GRU', 'LSTM', 'peepholeLSTM', 'residualLSTM'})} = 'LSTM'
    continueWithLearning (1,1) logical = false
    useMetaParPool (1,1) logical = true
    options.BayesObject (1,1) = false
    
    % Data options
    options.datasetPath (1,1) string = "trainingShuffled.mat"
    options.timewindowSize (1,1) double {mustBeInteger, mustBePositive} = 4
    
    % Bayes Optimization options
    options.GradientThresholdRange (1,2) double {mustBePositive} = [0.001 1000]
    options.NumberOfEpochsRange (1,2) double {mustBeInteger, mustBePositive} = [10 500]
    options.maxSeconds (1,1) double {mustBePositive} = Inf
    options.MaxObjectiveEvaluations (1,1) double {mustBeInteger, mustBePositive} = 100
    options.NumSeedPoints (1,1) double {mustBeInteger, mustBePositive} = 8
    options.doParallelBayes (1,1) logical = true
    options.SaveBayesObject (1,1) logical = true
    
    % Network options
    options.NetworkDepth (1,1) double {mustBeInteger, mustBePositive} = 1
    options.NumHiddenUnits (1,1) double {mustBeInteger, mustBePositive} = 100
    options.bidirectional (1,1) logical = false
    options.useDropout (1,1) logical = false
    options.useLayerNormalization (1,1) logical = false
    
    % Training options
    options.MiniBatchSize (1,1) double {mustBeInteger, mustBePositive} = 8
    options.useCrossValidation (1,1) logical = true
    options.crossValidationFolds (1,1) double {mustBeInteger, mustBePositive} = 2
    options.saveModels (1,1) logical = false
end

warningID = 'MATLAB:legend:IgnoringExtraEntries';
warning('off', warningID)

if useMetaParPool
    if (MetaParPool('open') <= 0)
        error('metacentrumGradientThreshold:MetaParPool', 'ERROR: Could not initialize MetaParPool');
    end % initializes parallel pool (returns the number of initialized workers)
end
optionsCells = namedargs2cell(options);

BayesObject = gradientThreshold(RNNType, continueWithLearning, optionsCells{:});

if useMetaParPool
    MetaParPool('close');       % closing the parallel pool
end
end