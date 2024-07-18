function BayesObject = lstmBayesOptimization(RNNType, options)
%LSTMBAYESOPTIMIZATION bayes optimization of LSTM network
%   arguments:
%       RNNType:
%           Type of the RNN used in each layer of network:
%               'GRU', 'LSTM', 'peepholeLSTM', 'residualLSTM'
%       options:
%           Optional arguments:
%               maxSeconds: maximum time in seconds for whole bayes optimization (default 24*60*60 = 1 day)
%               maxEpochs: maximum number of epochs for each training (default 100)
%               timewindowSize: size of the timewindow of the input data (default 4)
%               miniBatchSize: size of the minibatch (number of experiments in batch), default 8
%   returns:
%       BayesObject: bayes object with the results of the optimization
arguments
    RNNType (1,1) string {mustBeMember(RNNType, {'GRU', 'LSTM', 'peepholeLSTM', 'residualLSTM'})} = 'residualLSTM'
    options.maxSeconds (1,1) double {mustBeInteger, mustBePositive} = 24*60*60
    options.maxEpochs (1,1) double {mustBeInteger, mustBePositive} = 2
    options.timewindowSize (1,1) double {mustBeInteger, mustBePositive} = 4
    options.miniBatchSize (1,1) double {mustBeInteger, mustBePositive} = 8
end

optimVars = [
    %optimizableVariable('TimewindowSize', [4 8], 'Type', 'integer')
    optimizableVariable('NetworkDepth', [1 8], 'Type', 'integer')
    optimizableVariable('NumHiddenUnits', [10 500], 'Type', 'integer')
    optimizableVariable('InitialLearnRate', [1e-6 1e-4], 'Transform', 'log')
    %optimizableVariable('Momentum',[0.8 0.98]) % only with sgdm
    optimizableVariable('L2Regularization',[1e-10 1e-2],'Transform','log')
    optimizableVariable('bidirectional', {'true', 'false'}, 'Type', 'categorical')];

data = setupData( ...
    timewindowSize=options.timewindowSize, ...
    trainTestSplit = 0.95);

XTrain = data.XTrain;
YTrain = data.YTrain;
XTest = data.XTest;
YTest = data.YTest;

ObjFcn = makeObjFcn(XTrain, YTrain, XTest, YTest, RNNType, options.maxEpochs, options.timewindowSize, options.miniBatchSize);

BayesObject = bayesopt(ObjFcn, optimVars, ...
    'MaxTime', options.maxSeconds, ...
    'isObjectiveDeterministic', false, ... % with same data and same hyperparameters, the result of more trainNetwork function calls can be different
    'UseParallel', true);
end

function ObjFcn = makeObjFcn(XTrain, YTrain, XTest, YTest, RNNType, maxEpochs, timewindowSize, miniBatchSize)

ObjFcn = @valErrorFun;

    function [RMSEsum, cons, filePath] = valErrorFun(params)
        % in the experiment folder
        
        validationFrequency = floor(numel(XTrain)/miniBatchSize);
        
        inputSize = (timewindowSize-1)*6 + 2;
        outputSize = 6;
        net = setupNetwork( ...
            "inputSize", inputSize, ...
            "outputSize",outputSize, ...
            "bidirectional",strcmp(params.bidirectional, 'true'), ...
            "networkDepth", params.NetworkDepth, ...
            "numHiddenUnits", params.NumHiddenUnits, ...
            "RNNType", RNNType, ...
            "timewindowSize", timewindowSize, ...
            "layerGraph",true);
        
        options = trainingOptions( ...
            'adam', ...
            'Plots','training-progress', ...
            'Verbose',true, ...
            'VerboseFrequency',validationFrequency,...
            'MaxEpochs',maxEpochs,...
            'MiniBatchSize',miniBatchSize, ...
            'Shuffle','every-epoch', ...
            'ValidationData',{XTest,YTest}, ...
            'ValidationFrequency',validationFrequency, ...
            'ValidationPatience',Inf, ...
            'OutputNetwork','best-validation-loss', ...
            'InitialLearnRate', params.InitialLearnRate, ...
            'LearnRateSchedule','none',...
            'LearnRateDropPeriod',50,...
            'LearnRateDropFactor',0.1,...
            'L2Regularization',0.0001,...
            'GradientDecayFactor',0.9,...
            'GradientThreshold',10,...
            'ExecutionEnvironment', 'auto');
        
        % 'SquaredGradientDecayFactor', (0.9 RMSProp, 0.999 Adam)
        % 'CheckpointPath','checkpoints',...
        % 'CheckpointFrequency',10,...
        
        netDepth = "netdepth-" + params.NetworkDepth;
        nhid = "nhid-" + params.NumHiddenUnits;
        lr = "lr-" + params.InitialLearnRate;
        l2 = "l2-" + params.L2Regularization;
        bidir = "bidir-" + char(params.bidirectional);
        
        disp(netDepth + " " + nhid + " " + lr + " " + l2 + " " + bidir)
        
        trainedNet = trainNetwork(XTrain, YTrain, net, options);
        close(findall(groot,'Tag','NNET_CNN_TRAININGPLOT_UIFIGURE'))
        
        YPred = predict(trainedNet,XTest);
        % #TODO specify error function
        for n=1:numel(YTest)
            me = mean(YPred{n}(:) - YTest{n}(:));
            RMSE(n) = sqrt(me * me);
        end
        
        RMSEsum = sum(RMSE);
        curRMSEstr = num2str(RMSEsum);
        
        %boxplot(RMSE)
        %saveas(gcf, curRMSEstr + ".fig")
        
        fullFileName = mfilename("fullpath");
        [~, scriptName, ~] = fileparts(fullFileName);
        folderName = "results/" + scriptName + "/";
        if ~exist(folderName, 'dir')
            mkdir(folderName)
        end
        
        fileName = curRMSEstr + "_" + netDepth + "_" + nhid + "_" + lr + "_" + l2 + "_" + bidir + ".mat";
        filePath = folderName + fileName;
        save(filePath,'trainedNet','RMSE','options')
        cons = [];
        
    end
end