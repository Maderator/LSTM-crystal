function [lgraph, inputNames, outputName] = constructBiRNNLayer(numHiddenUnits,depth, options)
% CONSTRUCTBIRNNLAYER   Construct a bidirectional RNN layer
arguments
    numHiddenUnits (1,1) {mustBeNumeric, mustBePositive, mustBeInteger}
    depth (1,1) {mustBeNumeric, mustBePositive, mustBeInteger} = 1
    options.Name char = 'BiRNN'
    options.RNNType char {mustBeMember(options.RNNType, {'gru', 'lstm', 'peepholeLSTM', 'residualLSTM'})} = 'residualLSTM'
    options.OutputMode char {mustBeMember(options.OutputMode, {'last', 'sequence'})} = 'last'
    options.MergeMode char {mustBeMember(options.MergeMode, {'concatenation', 'sum', 'product', 'average', 'max'})} = 'concatenation'
end

rnnLayerFn = chooseRNNLayerClass(options.RNNType);

lgraph = layerGraph();

depthString = num2str(depth, "%d");
connectionLayerName = [options.Name, '_', options.RNNType, '_forward_', options.MergeMode, '_', depthString];
forwardRnnLayerName = [options.Name, '_', options.RNNType, '_forward_', depthString];
backwardRnnLayerName = [options.Name, '_', options.RNNType, '_backward_', depthString];
firstBackwardLayerName = ['flip_d', depthString, '_1'];
lastBackwardLayerName = backwardRnnLayerName;

layers_forward = [
    rnnLayerFn(numHiddenUnits, 'OutputMode', options.OutputMode, 'Name', forwardRnnLayerName)
    constructConnectionLayer(connectionLayerName, options.MergeMode)
    ];
lgraph = addLayers(lgraph, layers_forward);

layers_backward = [
    flipLayer(firstBackwardLayerName)
    rnnLayerFn(numHiddenUnits, 'OutputMode', options.OutputMode, 'Name', backwardRnnLayerName)
    ];
if options.OutputMode == "sequence"
    flip2Name = ['flip_d', depthString, '_2'];
    layers_backward = [layers_backward
        flipLayer(flip2Name)
        ];
    lastBackwardLayerName = flip2Name;
end
lgraph = addLayers(lgraph, layers_backward);

lgraph = connectLayers(lgraph, lastBackwardLayerName, [connectionLayerName,'/in2']);

inputNames = [string(forwardRnnLayerName), string(firstBackwardLayerName)];
outputName = connectionLayerName;

end

function connectionLayer = constructConnectionLayer(Name, MergeMode)
arguments
    Name char = ['BiRNN_concatenation'];
    MergeMode char = 'concatenation';
end

switch MergeMode
    case 'concatenation'
        connectionLayer = concatenationLayer(1, 2, 'Name', Name);
    case 'sum'
        connectionLayer = additionLayer(2, 'Name', Name);
    case 'product'
        connectionLayer = multiplicationLayer(2, 'Name', Name);
    case 'average'
        connectionLayer = averageLayer(2, 'Name', Name);
    case 'max'
        connectionLayer = maxLayer(2, 'Name', Name);
end
end