% Test inspired by Mathworks tutorial:
% "Sequence Classification Using Deep Learning"

% This test compares these two layers:
%   1. My custom LSTM layer implementation derived from Peephole LSTM Mathworks example tutorial
%   2. Mathworks LSTM layer

function hasExactWeights = compareCustomMatlabLSTM(epochs)
arguments
    epochs (1,1) = 2
end

addpath custom_layers\

miniBatchSize = 64;
numHiddenUnits = 100;

matlabLstmLayer = lstmLayer(numHiddenUnits, OutputMode="last");
myLstmLayer = customLSTMLayer(numHiddenUnits,OutputMode="last");

[preds, errors, nets] = trainAndTestNetworks(epochs, miniBatchSize, numHiddenUnits, myLstmLayer, matlabLstmLayer);


disp(sum(errors, 1))
disp(all(errors(:,1) == errors(:,2)))

matlabLstm = nets(2).Layers(2);

myLstm = nets(1).Layers(2);


hasExactWeights = compareWeights(matlabLstm, myLstm);
end

function hasExactWeights = compareWeights(lstm1, lstm2)
weights1 = getLstmWeights(lstm1);
weights2 = getLstmWeights(lstm2);
inW = isequal(weights1(1), weights2(1));
rW = isequal(weights1(2), weights2(2));
b = isequal(weights1(3), weights2(3));
hs = isequal(weights1(4), weights2(4));
cs = isequal(weights1(5), weights2(5));
if inW && rW && b && hs && cs
    hasExactWeights = 1;
    return
else
    disp(inW + " " + rW + " "  + b + " " + hs + " " + cs)
    disp("Absolute difference:")
    disp("Inputweights: " + sum(abs(weights1(1) - weights2(1))))
    disp("RecurrentWeights: " + sum(abs(weights1(2) - weights2(2))))
    disp("Bias: " + sum(abs(weights1(3) - weights2(3))))
    disp("HiddenState: " + sum(abs(weights1(4) - weights2(4))))
    disp("CellState: " + sum(abs(weights1(5) - weights2(5))))
    disp("Sum of absolute values of weights:")
    disp("Inputweights " + sum(abs(weights1(1))) + " " + sum(abs(weights2(1))))
    disp("RecurrentWeights " + sum(abs(weights1(2))) + " " + sum(abs(weights2(2))))
    disp("Bias " + sum(abs(weights1(3))) + " " + sum(abs(weights2(3))))
    disp("HiddenState " + sum(abs(weights1(4))) + " " + sum(abs(weights2(4))))
    disp("CellState " + sum(abs(weights1(5))) + " " + sum(abs(weights2(5))))
end

hasExactWeights = 0;
end

function [inW, rW, b, hs, cs] = getLstmWeights(lstm)
inW = lstm.InputWeights;
rW = lstm.RecurrentWeights;
b = lstm.Bias;
hs = lstm.HiddenState;
cs = lstm.CellState;
end

function [preds, errors, nets] = trainAndTestNetworks(epochs, miniBatchSize, numHiddenUnits, layer1, layer2)
arguments
    epochs (1,1) = 1
    miniBatchSize = 64;
    numHiddenUnits = 100;
    layer1 = customLSTMLayer(numHiddenUnits, OutputMode="last");
    layer2 = lstmLayer(numHiddenUnits, OutputMode="last");
end

[XTrain, YTrain] = japaneseVowelsTrainData();
[XTrain, YTrain] = prepareSequenceData(XTrain, YTrain);

[XTest, YTest] = japaneseVowelsTestData();
[XTest, YTest] = prepareSequenceData(XTest, YTest);

rng('default')
net1 = trainLSTMnetwork(XTrain, YTrain, miniBatchSize, layer1, epochs);
rng('default')
net2 = trainLSTMnetwork(XTrain, YTrain, miniBatchSize, layer2, epochs);

myYPred = testNetwork(net1, XTest, YTest, miniBatchSize);
matlabYPred = testNetwork(net2, XTest, YTest, miniBatchSize);
myErrors = myYPred ~= YTest;
matlabErrors = matlabYPred ~= YTest;

preds = [myYPred, matlabYPred];
errors = [myErrors, matlabErrors];
nets = [net1, net2];
end

function [X, y, sequenceLengths] = prepareSequenceData(X, y)
numObservations = numel(X);
for i=1:numObservations
    sequence = X{i};
    sequenceLengths(i) = size(sequence,2);
end

[sequenceLengths,idx] = sort(sequenceLengths);
X = X(idx);
y = y(idx);
end

function net = trainLSTMnetwork(XTrain, YTrain, miniBatchSize, usedLstmLayer, epochs)
inputSize = 12;
numClasses = 9;

layers = [ ...
    sequenceInputLayer(inputSize)
    usedLstmLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

options = trainingOptions("adam", ...
    ExecutionEnvironment="cpu", ...
    GradientThreshold=1, ...
    MaxEpochs=epochs, ...
    MiniBatchSize=miniBatchSize, ...
    SequenceLength="longest", ...
    Shuffle="never", ...
    Verbose=0, ...
    Plots="training-progress");

net = trainNetwork(XTrain,YTrain,layers,options);
end

function YPred = testNetwork(net, XTest, ~, miniBatchSize)
YPred = classify(net,XTest, ...
    MiniBatchSize=miniBatchSize, ...
    SequenceLength="longest");
end