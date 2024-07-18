function [valError, trainingTime] = trainMyRNNNetwork(RNNType, options)
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
    
    % Data options
    options.datasetPath (1,1) string = "trainingShuffled.mat"
    options.timewindowSize (1,1) double {mustBeInteger, mustBePositive} = 4
    
    % Network options
    options.NetworkDepth (1,1) double {mustBeInteger, mustBePositive} = 1
    options.NumHiddenUnits (1,1) double {mustBeInteger, mustBePositive} = 100
    options.bidirectional (1,1) logical = false
    options.useDropout (1,1) logical = false
    options.useLayerNormalization (1,1) logical = false
    options.useResidualConnection (1,1) logical = false
    
    % Training options
    options.Repetitions (1,1) double {mustBeInteger, mustBePositive} = 1
    options.NumberOfEpochs (1,1) double {mustBeInteger, mustBePositive} = 1
    options.MiniBatchSize (1,1) double {mustBeInteger, mustBePositive} = 8
    options.useCrossValidation (1,1) logical = false
    options.crossValidationFolds (1,1) double {mustBeInteger, mustBePositive} = 2
    options.GradientThreshold (1,1) double {mustBePositive} = 2
    options.saveModels (1,1) logical = false
end

data = setupData( ...
    datasetPath=options.datasetPath, ...
    timewindowSize=options.timewindowSize, ...
    useTrainTestSplit=~options.useCrossValidation, ...
    trainTestSplit = 0.9);

XTrain = data.XTrain;
YTrain = data.YTrain;
if ~options.useCrossValidation
    XTest = data.XTest;
    YTest = data.YTest;
end

if options.useCrossValidation
    trainTestRatio = (options.crossValidationFolds-1) / options.crossValidationFolds;
    validationFrequency = floor(numel(XTrain)*trainTestRatio/options.MiniBatchSize);
else
    validationFrequency = floor(numel(XTrain)/options.MiniBatchSize);
end

inputSize = (options.timewindowSize-1)*6 + 2;
outputSize = 6;

net = setupNetwork( ...
    "inputSize", inputSize, ...
    "outputSize",outputSize, ...
    "bidirectional",options.bidirectional, ...
    "networkDepth", options.NetworkDepth, ...
    "numHiddenUnits", options.NumHiddenUnits, ...
    "RNNType", RNNType, ...
    "layerGraph",true, ...
    "useDropout", options.useDropout, ...
    "useLayerNormalization", options.useLayerNormalization, ...
    "useResidualConnections",  false);

if options.useCrossValidation
    cvp = cvpartition(numel(XTrain),'KFold',options.crossValidationFolds);
    Repetitions = options.crossValidationFolds;
else
    Repetitions = options.Repetitions;
end

RMSEsums = zeros(1, Repetitions);
trainingTimes = zeros(1, Repetitions);

for repNum = 1:Repetitions
    disp("Starting repetition " + repNum)
    if options.useCrossValidation
        XTrainPart = XTrain(training(cvp, repNum));
        YTrainPart = YTrain(training(cvp, repNum));
        XTest = XTrain(test(cvp, repNum));
        YTest = YTrain(test(cvp, repNum));
    else
        XTrainPart = XTrain;
        YTrainPart = YTrain;
    end
    
    trainOptions = trainingOptions( ...
        'adam', ...
        'Verbose',false, ...
        'VerboseFrequency',validationFrequency,...
        'MaxEpochs',options.NumberOfEpochs,...
        'MiniBatchSize',options.MiniBatchSize, ...
        'Shuffle','every-epoch', ...
        'ValidationData',{XTest,YTest}, ...
        'ValidationFrequency',validationFrequency, ...
        'ValidationPatience',Inf, ...
        'OutputNetwork','best-validation-loss', ...
        'InitialLearnRate', 0.001, ...
        'LearnRateSchedule','none',...
        'LearnRateDropPeriod',50,...
        'LearnRateDropFactor',0.1,...
        'L2Regularization',0.0001,...
        'GradientDecayFactor',0.9,...
        'GradientThreshold',options.GradientThreshold,...
        'ExecutionEnvironment', 'auto');
    
    tic
    trainedNet = trainNetwork(XTrainPart, YTrainPart, net, trainOptions);
    trainingTimes(repNum) = toc;
    close(findall(groot,'Tag','NNET_CNN_TRAININGPLOT_UIFIGURE'))
    
    YPred = predict(trainedNet,XTest);
    RMSE = zeros(1, numel(YTest));
    for n=1:numel(YTest)
        RMSE(n) = rmse(YPred{n}(:), YTest{n}(:));
    end
    RMSEsums(repNum) = sum(RMSE);
    curRMSEstr = num2str(RMSEsums(repNum));
    
    folderName = "results/" + RNNType + "/";
    if ~exist(folderName, 'dir')
        mkdir(folderName)
    end
    
    fileName = curRMSEstr + "_" + RNNType + ".mat";
    filePath = folderName + fileName;
    if options.saveModels
        save(filePath,'trainedNet','RMSE','options')
    end
end
valError = mean(RMSEsums);
trainingTime = mean(trainingTimes);

end