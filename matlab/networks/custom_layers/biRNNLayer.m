
classdef biRNNLayer < nnet.layer.Layer ...
        & nnet.layer.Formattable ...
        & nnet.layer.Acceleratable
    %biRNNLayer bidirectional RNN Layer
    
    properties
        NumHiddenUnits
        OutputMode
        MergeMode
    end
    
    properties (Learnable, State)
        % Nested dlnetwork objects with both learnable
        % parameters and state parameters.
        Network
    end
    
    methods
        function layer = biRNNLayer(numHiddenUnits, rnnType, args)
            %biRNNLayer bidirectional RNN Layer.
            %   Arguments:
            %       numHiddenUnits - Number of hidden units of the RNN.
            %       Number of hidden units in one half of bidirectional
            %       RNN.
            %       rnnType        - Type of RNN. One of the following:
            %           'GRU', 'LSTM', 'peepholeLSTM', 'residualLSTM'
            %   layer = customLSTMLayer(numHiddenUnits, rnnType)
            %   creates a bidirectional RNN layer with the specified number of
            %   hidden units and specified type of RNN.
            %
            %   layer = customLSTMLayer(numHiddenUnits, rnnType, Name=Value)
            %   creates a LSTM layer and specifies additional
            %   options using one or more name-value arguments:
            %
            %   Name       - Name of the layer, specified as a string.
            %                The default is "biRNN".
            %
            %   OutputMode - Output mode, specified as one of the
            %                following:
            %                "sequence" - Output the entire sequence
            %                             of data.
            %
            %                 "last"     - Output the last time step
            %                              of the data.
            %                 The default is "sequence".
            %   MergeMode - Mode of merging the outputs of the forward and backward layers.
            %               One of the following: 'concatenation', 'sum', 'product', 'average', 'max'.
            %               The default is 'concatenation'.
            
            % Parse input arguments.
            arguments
                numHiddenUnits (1,1) {mustBeInteger, mustBePositive};
                rnnType string {mustBeMember(rnnType,{'GRU', 'LSTM', 'peepholeLSTM', 'residualLSTM'})};
                args.Name (1,1) string = "biRNN";
                args.OutputMode (1,1) string {mustBeMember(args.OutputMode,{'sequence', 'last'})} = "sequence";
                args.MergeMode (1,1) string {mustBeMember(args.MergeMode,{'concatenation', 'sum', 'product', 'average', 'max'})} = "concatenation";
            end
            
            layer.NumHiddenUnits = numHiddenUnits;
            layer.Name = args.Name;
            layer.OutputMode = args.OutputMode;
            layer.MergeMode = args.MergeMode;
            layer.Description = "Bidirectional RNN of type " + rnnType + " with " + numHiddenUnits + " hidden units";
            layer.Type = "Bidirectional " + rnnType;
            
            % Create connection layer.
            rnnLayerFn = chooseRNNLayerClass(rnnType);
            connectionName = rnnType + args.MergeMode;
            connectionLayer = layer.constructConnectionLayer(connectionName, args.MergeMode);
            
            % Forward layer
            forwardName = rnnType + "_forward";
            layersForward = [
                rnnLayerFn(numHiddenUnits, 'OutputMode', args.OutputMode, 'Name', forwardName)
                connectionLayer
                ];
            
            % backward layer
            backwardName = rnnType + "_backward";
            layersBackward = [
                flipLayer(rnnType + "_flip1")
                rnnLayerFn(numHiddenUnits, 'OutputMode', args.OutputMode, 'Name', backwardName)
                ];
            
            % sequence has to be flipped back again to match forward
            if args.OutputMode == "sequence"
                flip2Name = rnnType + "_flip2";
                layersBackward = [
                    layersBackward
                    flipLayer(flip2Name)
                    ];
                backwardName = flip2Name;
            end
            
            lgraph = layerGraph(layersForward);
            lgraph = addLayers(lgraph, layersBackward);
            
            lgraph = connectLayers(lgraph, backwardName, connectionName + "/in2");
            
            net = dlnetwork(lgraph, Initialize=false);
            %plot(layerGraph(net))
            
            layer.Network = net;
        end
        
        function [Z,state] = predict(layer,X)
            net = layer.Network;
            [Z,state] = predict(net, X, X); % We have two inputs, one for forward and one for backward layer.
        end
        
        function layer = resetState(layer)
            layer.Network = resetState(layer.Network);
        end
    end
    
    methods (Access=protected)
        function connectionLayer = constructConnectionLayer(layer, Name, MergeMode)
            arguments
                layer;
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
    end
end

