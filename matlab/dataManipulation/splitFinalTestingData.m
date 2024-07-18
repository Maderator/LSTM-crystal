%% Load raw data
RawData = load("sequencesInputsOutputs.mat");
data.Inputs = RawData.InputsAll;
data.Outputs = RawData.OutputsAll;


%% split to shuffled/sorted data for bayes optimization and for final testing

% there are 500 experiments in total
%   -> we choose 450 for optimization and 50 for final testing
[noShuffleBayesOpt, noShuffleFinalTesting] = splitData(data, ...
    trainPartAbsolute=450, ...
    shuffle=false);

[shuffleBayesOpt, shuffleFinalTesting] = splitData(data, ...
    trainPartAbsolute=450, ...
    shuffle=true, ...
    rng="default");

%% save data
data = noShuffleBayesOpt;
saveInputsOutputs(data, "trainingSorted.mat");

data = noShuffleFinalTesting;
saveInputsOutputs(data, "finalTestingSorted.mat");

data = shuffleBayesOpt;
saveInputsOutputs(data, "trainingShuffled.mat");

data = shuffleFinalTesting;
saveInputsOutputs(data, "finalTestingShuffled.mat");

function saveInputsOutputs(data, path)
Inputs = data.Inputs;
Outputs = data.Outputs;
save(path, "Inputs", "Outputs")
end


