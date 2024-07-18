function ObjFcn = makeObjectiveFunction(data, timewindowSize, RNNType, options)

% data contains:
%   only XTrain, YTrain when options.useCrossValidation is true
%   XTrain, YTrain, XTest, YTest when options.useCrossValidation is false

XTrain = data.XTrain;
YTrain = data.YTrain;
if ~options.useCrossValidation
    XTest = data.XTest;
    YTest = data.YTest;
end

ObjFcn = @valErrorFun;

    function [valError, cons, filePath] = valErrorFun(params)
        
        % set validation frequency, input size, and output size
        if options.useCrossValidation
            trainTestRatio = (options.crossValidationFolds-1) / options.crossValidationFolds;
            validationFrequency = floor(numel(XTrain)*trainTestRatio/options.MiniBatchSize);
        else
            validationFrequency = floor(numel(XTrain)/options.MiniBatchSize);
        end
        
        inputSize = (timewindowSize-1)*6 + 2;
        outputSize = 6;
        
        % setup network and training options
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
        
        % 'SquaredGradientDecayFactor', (0.9 RMSProp, 0.999 Adam)
        % 'CheckpointPath','checkpoints',...
        % 'CheckpointFrequency',10,...
        
        epoch = "ep-" + params.NumberOfEpochs;
        grad = "grad-" + params.GradientThreshold;
        infoString = grad + "_" + epoch;
        
        %disp(infoString)
        
        if options.useCrossValidation
            cvp = cvpartition(numel(XTrain),'KFold',options.crossValidationFolds);
            Repetitions = options.crossValidationFolds;
        else
            Repetitions = 1;
        end
        
        RMSEsums = zeros(1, Repetitions);
        for repNum = 1:Repetitions
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
                'Plots','training-progress', ...
                'Verbose',true, ...
                'VerboseFrequency',validationFrequency,...
                'MaxEpochs',params.NumberOfEpochs,...
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
                'GradientThreshold',params.GradientThreshold,...
                'ExecutionEnvironment', 'auto');
            
            trainedNet = trainNetwork(XTrainPart, YTrainPart, net, trainOptions);
            close(findall(groot,'Tag','NNET_CNN_TRAININGPLOT_UIFIGURE'))
            
            YPred = predict(trainedNet,XTest);
            % #TODO specify error function
            RMSE = zeros(1, numel(YTest));
            for n=1:numel(YTest)
                RMSE(n) = rmse(YPred{n}(:), YTest{n}(:));
            end
            RMSEsums(repNum) = sum(RMSE);
            curRMSEstr = num2str(RMSEsums(repNum));
            
            %boxplot(RMSE)
            %saveas(gcf, curRMSEstr + ".fig")
            
            % #TODO encapsualte this to if statement and add option to not
            % save the mat file with trained net, rmse, and options
            
            folderName = "results/" + RNNType + "/";
            if ~exist(folderName, 'dir')
                mkdir(folderName)
            end
            
            fileName = curRMSEstr + "_" + infoString + ".mat";
            filePath = folderName + fileName;
            if options.saveModels
                save(filePath,'trainedNet','RMSE','options')
            end
            cons = [];
        end
        valError = mean(RMSEsums);
    end

end
