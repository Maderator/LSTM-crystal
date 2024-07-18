function net = setupNetwork(options)
%SETUPNETWORK Construct a network based on the given parameters
% Returns dlnetowrk type
arguments
    options.bidirectional (1,1) logical = true
    options.bidirectionalMergeMode (1,1) string {mustBeMember(options.bidirectionalMergeMode,{'concatenation', 'sum', 'product', 'average', 'max'})} = "concatenation";
    options.dropoutRate (1,1) double {mustBeInRange(options.dropoutRate, 0, 1)} = 0.2
    options.inputSize (1,1) double {mustBeInteger, mustBePositive} = 20
    options.layerGraph (1,1) logical = false % If true, the function returns a layerGraph instead of a dlnetwork
    options.networkDepth (1,1) double {mustBeInteger, mustBePositive} = 1
    options.numHiddenUnits (1,1) double {mustBeInteger, mustBePositive} = 20
    options.outputMode (1,1) string {mustBeMember(options.outputMode, {'last', 'sequence'})} = 'sequence'
    options.outputSize (1,1) double {mustBeInteger, mustBePositive} = 6
    options.plotNetwork (1,1) logical = false
    options.RNNType (1,1) string {mustBeMember(options.RNNType, {'GRU', 'LSTM', 'peepholeLSTM', 'residualLSTM'})} = 'residualLSTM'
    options.useDropout (1,1) logical = false
    options.useLayerNormalization (1,1) logical = false
    options.useResidualConnections (1,1) logical = false
end

layerClass = chooseRNNLayerClass(options.RNNType);

lgraph = layerGraph();

[lgraph, previousLayerName] = addInputLayers(lgraph, options.inputSize);

for depth=1:options.networkDepth
    % last layer output mode
    if depth == options.networkDepth
        outputMode = options.outputMode;
    else
        outputMode = 'sequence';
    end
    
    if options.bidirectional
        currentLayerName = "d" + depth + "_bidirectional_" + options.RNNType;
        layers = biRNNLayer(options.numHiddenUnits, options.RNNType, 'Name', currentLayerName, 'outputMode', outputMode, 'MergeMode', options.bidirectionalMergeMode);
    else
        currentLayerName = sprintf('d%d_%s', depth, options.RNNType);
        layers = layerClass(options.numHiddenUnits, ...
            Name=currentLayerName, ...
            OutputMode=outputMode ...
            );
        
    end
    lgraph = addLayers(lgraph, layers);
    lgraph = connectLayers(lgraph, previousLayerName, currentLayerName);
    
    if options.useDropout
        dropoutLayerName = "d" + depth + "_dropout";
        lgraph = addLayers(lgraph, dropoutLayer(0.2, 'Name', dropoutLayerName));
        lgraph = connectLayers(lgraph, currentLayerName, dropoutLayerName);
        currentLayerName = dropoutLayerName;
    end
    
    if options.useResidualConnections
        residualConnectionLayerName = "d" + depth + "_residualConnection";
        layers = [];
        if depth == 1
            layers = [layers convolution1dLayer(1, options.numHiddenUnits, 'Name', residualConnectionLayerName + '_conv1d')];
        end
        layers = [layers additionLayer(2, 'Name', residualConnectionLayerName + '_add')];
        lgraph = addLayers(lgraph, layers);
        if depth == 1
            lgraph = connectLayers(lgraph, previousLayerName, residualConnectionLayerName + '_conv1d');
        else
            lgraph = connectLayers(lgraph, previousLayerName, residualConnectionLayerName+ "_add/in1");
        end
        lgraph = connectLayers(lgraph, currentLayerName, residualConnectionLayerName+ "_add/in2");
        currentLayerName = residualConnectionLayerName + "_add";
    end
    
    if options.useLayerNormalization
        layerNormalizationLayerName = "d" + depth + "_layerNormalization";
        lgraph = addLayers(lgraph, layerNormalizationLayer('Name', layerNormalizationLayerName));
        lgraph = connectLayers(lgraph, currentLayerName, layerNormalizationLayerName);
        currentLayerName = layerNormalizationLayerName;
    end
    
    previousLayerName = currentLayerName;
end

[lgraph, outputLayerName] = addOutputLayers(lgraph, options.outputSize, options.layerGraph);
lgraph = connectLayers(lgraph, previousLayerName, outputLayerName);

if options.plotNetwork
    plot(lgraph);
end

if options.layerGraph
    net = lgraph;
else
    net = dlnetwork(lgraph);
end

end

function [lgraph, lastLayerName] = addInputLayers(lgraph, inputSize)
%ADDINPUTLAYERS Add input layers to the layer graph
%   Adds the following layers to the layer graph:
%   - sequenceInputLayer

lgraph = addLayers(lgraph, sequenceInputLayer(inputSize, 'Name', 'input'));
lastLayerName = 'input';

end

function [lgraph, firstLayerName] = addOutputLayers(lgraph, outputSize, adddOuputLayer)
%ADDOUTPUTLAYERS Add output layers to the layer graph
%   Adds the following layers to the layer graph:
%   - fullyConnectedLayer
%   - regressionLayer if adddOuputLayer is true else regressionLayer is not added

firstLayerName = 'fullyConnected';
layers = [fullyConnectedLayer(outputSize, 'Name', firstLayerName)];
if adddOuputLayer
    layers = [layers regressionLayer('Name', 'RegressionOutput')];
end
lgraph = addLayers(lgraph, layers);

end