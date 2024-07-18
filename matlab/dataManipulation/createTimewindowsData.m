function dataTimewindows = createTimewindowsData(data, windowSize)
%CREATETIMEWINDOWS Create sliding windows from given data with given
%   Given the windowSize, create sliding windows dataset with the window
%   size.
%   Inputs:
%       data - Inputs and Outputs data with dimensions [simulation,timesteps,datapoints]
%   Returns:
%       dataTimewindows - Inputs and Outputs data with dimensions
%                         [simulation, timesteps, windows, datapoints]
arguments
    data
    windowSize = 4;
end
% #BUG the outputs should be shifted in regards to input time windows so that the output window starts in the next time step after the input ends
Inputs = data.Inputs;
Outputs = data.Outputs;

dataTimewindows.Inputs = createSingleTimewindow(Inputs, windowSize);
dataTimewindows.Outputs = createSingleTimewindow(Outputs, windowSize);

end

function dataTimewindows = createSingleTimewindow(data, windowSize)
%CREATESINGLETIMEWINDOW Create timewindow from either Inputs or Outputs
dataSize = size(data);
timedimensionSize = dataSize(2);
dataTimewindows = ...
    zeros( ...
    windowSize, ...
    dataSize(1), ...
    dataSize(2) - (windowSize-1), ...
    dataSize(3)...
    );
for n=1:windowSize
    timestep = data(:,n:timedimensionSize-(windowSize - n), :);
    dataTimewindows(n,:,:,:) = timestep;
end

dataTimewindows = permute(dataTimewindows, [2 3 1 4]);
end