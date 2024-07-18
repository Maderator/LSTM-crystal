exampleParams.NetworkDepth = 1;
exampleParams.NumHiddenUnits = 20;
exampleParams.InitialLearnRate = 0.001;
exampleParams.L2Regularization = 0.001;
exampleParams.bidirectional = "true";

maxEpochs = 2;
timewindowSize = 4;
miniBatchSize = 16;

optimVars = [
    %optimizableVariable('TimewindowSize', [4 8], 'Type', 'integer')
    optimizableVariable('NetworkDepth', [1 8], 'Type', 'integer')
    optimizableVariable('NumHiddenUnits', [10 500], 'Type', 'integer')
    optimizableVariable('InitialLearnRate', [1e-5 1], 'Transform', 'log')
    %optimizableVariable('Momentum',[0.8 0.98]) % only with sgdm
    optimizableVariable('L2Regularization',[1e-10 1e-2],'Transform','log')
    optimizableVariable('bidirectional', {'true', 'false'}, 'Type', 'categorical')];

data = setupData( ...
    timewindowSize=timewindowSize, ...
    trainTestSplit = 0.95);

XTrain = data.XTrain;
YTrain = data.YTrain;
XTest = data.XTest;
YTest = data.YTest;

ObjFcn = makeObjFcn(XTrain, YTrain, XTest, YTest, maxEpochs, timewindowSize, miniBatchSize);

%ObjFcn(exampleParams)
%% Optimization

% #TODO choose parameters
BayesObject = bayesopt(ObjFcn, optimVars, ...
    'MaxTime', 24*60*60, ...
    'MaxObjectiveEvaluations',100,...
    'isObjectiveDeterministic', false, ...
    'UseParallel', true);

%% Final evaluation

bestIdx = BayesObject.IndexOfMinimumTrace(end);
fileName = BayesObject.UserDataTrace{bestIdx};
savedStruct = load(fileName);
valError = savedStruct.RMSE

%%

YPred = predict(savedStruct.trainedNet,XTest);
for n=1:numel(YTest)
    me = mean(YPred{n}(:) - YTest{n}(:));
    RMSE(n) = sqrt(me * me);
end

RMSEsum = sum(RMSE);
testError = RMSEsum;

%% Objective function
function ObjFcn = makeObjFcn(XTrain, YTrain, XTest, YTest, maxEpochs, timewindowSize, miniBatchSize)

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
            "RNNType", "LSTM", ...
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