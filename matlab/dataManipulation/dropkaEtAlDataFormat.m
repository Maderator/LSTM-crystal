function preparedData = dropkaEtAlDataFormat(data, timestepsNum, featuresFirst)
%DROPKAETALDATAFORMAT prepare Input and Output data as in [Dropka et al.]
% Dropka et al. use in "Real Time Predictions of VGF-GaAs Growth Dynamics
% by LSTM Neural Networks" last input and n last outputs from previous
% time steps as input to the LSTM network and as output of network they use
% output (temperature measurements and solid-liquid surface position)
% at current timestep.
%   Inputs:
%       data.Inputs - Input data with dimensions [simulationsNum,
%       historyLen, inputFeaturesNum]
%       data.Outputs - Output data with dimensions [simulationsNum,
%       historyLen, outputFeaturesNum]
%       timestepsNum - number of timesteps history to take into account
%       when creating inputs for neural network
%       featuresFirst - If true, the dimensions are ordered with features
%       being first and time being second dimension
%   Outputs:
%       preparedData.Inputs - Cell array of prepared experiments for
%       input to neural network
%       preparedData.Outputs - Cell array of prepared outputs of
%       experiments expected as output of neural network.
%   We assume that Inputs and Outputs have same number of
%   experiments (500) and timesteps (100)
arguments
    data (1,1) struct
    timestepsNum (1,1) double {mustBeInteger, mustBePositive} = 4;
    featuresFirst (1,1) logical = true;
end

Inputs = data.Inputs;
Outputs = data.Outputs;

inputsSize = size(Inputs);
experimentsNum = inputsSize(1);
historyLen = inputsSize(2);

inputFeaturesNum = inputsSize(3);
outputFeaturesNum = size(Outputs, 3);
% Networks inputs consist of Inputs (heat fluxes) from current time step
% and Outputs (temperature sensors + position of the solid-liquid
% interface) from last "timestepsNum-1" time steps
featuresNum = inputFeaturesNum + outputFeaturesNum * (timestepsNum-1);

networkInputs = zeros(...
    experimentsNum, ...
    historyLen - timestepsNum + 1, ...
    featuresNum ...
    );

% get last timestep Inputs
selectedInputs = Inputs(:,timestepsNum:end,:);
networkInputs(:,:,1:2) = selectedInputs;

% get last timestepsNum-1 Outputs before current timestep
for n=0:timestepsNum-2
    selectedOutputs = Outputs(:,timestepsNum-(n+1):historyLen-(n+1),:);
    featuresOffset = 3 + n*outputFeaturesNum;
    networkInputs(:,:,featuresOffset:featuresOffset+outputFeaturesNum-1) = selectedOutputs;
end

% get network outputs
networkOutputs = Outputs(:,timestepsNum:end,:);

% pre allocate cell arrays
preparedData.Inputs = cell(1, experimentsNum);
preparedData.Outputs = cell(1, experimentsNum);

% populate cell array with inputs and outputs data
for n=1:experimentsNum
    inputs = reshape(networkInputs(n,:,:), size(networkInputs, [2 3]));
    outputs = reshape(networkOutputs(n,:,:), size(networkOutputs, [2, 3]));
    if featuresFirst
        inputs = inputs.';
        outputs = outputs.';
    end
    preparedData.Inputs{n} = inputs;
    preparedData.Outputs{n} = outputs;
end