function normalizedData = normalizeData(data)
%NORMALIZEDATA normalize data
arguments
    data (1,1) struct
end

normalizedData.Inputs = normalizeInputs(data.Inputs);
normalizedData.Outputs = normalizeOutputs(data.Outputs);
assert(...
    allBetweenZeroAndOne(normalizedData.Inputs) && ...
    allBetweenZeroAndOne(normalizedData.Outputs), ...
    "Normalization failed! Inputs are not between zero and one.")

end

function normIn = normalizeInputs(Inputs)
%NORMALIZE_INPUTS normalize fluxes
% inputs consist of 2 fluxes

[minFlux, maxFlux] = getDataMinMax(Inputs);
deltaFlux = maxFlux - minFlux;
normIn = (Inputs-minFlux)/deltaFlux;

end

function normOut = normalizeOutputs(Outputs)
%NORMALIZE_OUTPUTS normalize temperatures and position
% Outputs consist of 5 temperature measures and 1 position of the
% solid-liquid interface

[minTemp, maxTemp] = getDataMinMax(Outputs(:,:,1:5));
deltaTemp = maxTemp - minTemp;
[minPos, maxPos] = getDataMinMax(Outputs(:,:,6));
deltaPos = maxPos - minPos;
normOut(:,:,1:5) = (Outputs(:,:,1:5) - minTemp)/deltaTemp;
normOut(:,:,6) = (Outputs(:,:,6) - minPos)/deltaPos;

end

function [dataMin, dataMax] = getDataMinMax(data)
%GETDATAMINMAX get minimum and maximum scalar value of data

dataMin = min(data, [], 'all');
dataMax = max(data, [], 'all');

end

function isTrue = allBetweenZeroAndOne(X)
%ALLBETWEENZEROANDONE True if all elements are between one and zero

isTrue = all(X >= 0, 'all') && all(X <= 1, 'all');

end