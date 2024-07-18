function [train, test] = splitData(data, options)
% SPLITDATA splits the sequencesINputsOutputs to two sets for optimization and final validation
% Inputs:
%   data: struct with data.Inputs and data.Outputs
% Optional inputs:
%   trainPartSize: the size of the train set in percent.
%                        If subsequent size of train set is not whole number,
%                        a closest lower integer is taken for size of train
%                        set. (default is 0.9)
%   trainPartAbsoluteSize: if not zero, this value defines size of train set.
%                           (default 0)
%   shuffle: If true, shuffle data (default is true)
%   rng: random number generator options (default is "default")
% Outputs:
%   train: struct with train.Inputs and train.Outputs
%   test: struct with test.Inputs and test.Outputs
arguments
    data (1,1) struct;
    options.trainPartSize (1,1) double {mustBeNonnegative, mustBeLessThanOrEqual(options.trainPartSize, 1)} = 0.9;
    options.trainPartAbsolute (1,1) double {mustBeInteger, mustBeNonnegative} = 0;
    options.shuffle (1,1) logical = true;
    options.rng (1,1) string = "default";
end

sequencesNum = size(data.Inputs, 2);

if options.trainPartAbsolute
    trainSize = options.trainPartAbsolute;
else
    trainSize = floor(sequencesNum * options.trainPartSize);
end

if options.shuffle
    rng(options.rng)
    permutation = randperm(sequencesNum);
    trainIdxs = permutation(1:trainSize);
    testIdxs = permutation(trainSize+1:sequencesNum);
else
    trainIdxs = 1:trainSize;
    testIdxs = trainSize+1:sequencesNum;
end

train.Inputs = data.Inputs(trainIdxs);
test.Inputs = data.Inputs(testIdxs);

train.Outputs = data.Outputs(trainIdxs);
test.Outputs = data.Outputs(testIdxs);
end