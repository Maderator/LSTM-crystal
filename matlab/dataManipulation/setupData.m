function data = setupData(options)
%SETUPDATA Setup data for experimentation or evaluation
% Optional Inputs:
%   options: struct with fields:
%       datasetPath: path to .mat file with Inputs and Outputs variables
%       timewindowSize: size of time window
%       useDropkaEtAl: boolean, if true, use Dropka et al. methodology for data preparation
%       useTrainTestSplit: boolean, if true, split data into train and test sets
%       trainTestSplit: double, if useTrainTestSplit is true, specifies the size of train set
%       shuffleData: boolean, if true, shuffle data before splitting into train and test sets
%       seed: 0 | positive integer | 'default' (default) | 'shuffle', seed for random number generator
% Outputs:
%   data: struct with fields:
%       if useTrainTestSplit is true:
%           XTrain: cell array of sequences of input features for training
%           YTrain: cell array of sequences of output features for training
%           XTest: cell array of sequences of input features for testing
%           YTest: cell array of sequences of output features for testing
%       else:
%           Inputs: 4d array of time windows of input features
%           Outputs: 4d array of time windows of output features

arguments
    options.datasetPath (1,1) string = "trainingShuffled.mat";
    options.timewindowSize (1,1) double {mustBeInteger,mustBePositive} = 4;
    options.useDropkaEtAl (1,1) logical = true;
    options.useTrainTestSplit (1,1) logical = true;
    options.trainTestSplit (1,1) double {mustBeInRange(options.trainTestSplit,0,1)} = 0.9;
    options.shuffleData (1,1) logical = false;
    options.seed = "default";
end


data = load(options.datasetPath);

% Normalize data
data = normalizeData(data);

% Create time windows
if options.useDropkaEtAl
    % create cell arrays of input features and output features in cell arrays as
    % specified in Dropka et al. paper.
    % The usage of cell array of sequences is specified in trainNetwork function
    % documentation (https://www.mathworks.com/help/deeplearning/ref/trainnetwork.html?s_tid=doc_ta#mw_36a68d96-8505-4b8d-b338-44e1efa9cc5e)
    preparedData = dropkaEtAlDataFormat(data, options.timewindowSize);
else
    % create plain 4d array of time windows
    error("Only Dropka et al. methodology is 100 % functional for now")
    %preparedData = createTimewindowsData(data, options.timewindowSize);
end

% train test split of data
if options.useTrainTestSplit
    [train, test] = splitData(preparedData, ...
        trainPartSize=options.trainTestSplit, ...
        shuffle=options.shuffleData, ...
        rng=options.seed);
    
    data = struct;
    data.XTrain = train.Inputs;
    data.YTrain = train.Outputs;
    data.XTest = test.Inputs;
    data.YTest = test.Outputs;
else
    data = struct;
    data.XTrain = preparedData.Inputs;
    data.YTrain = preparedData.Outputs;
end