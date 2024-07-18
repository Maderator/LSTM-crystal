% residualLSTMLayer is based on custmLSTMLayer.m which is derived from the Mathworks
% tutorial Define Custom Recurrent Deep Learning Layer

classdef residualLSTMLayer < nnet.layer.Layer ...
        & nnet.layer.Formattable
    %RESIDUALLSTMLAYER Version of LSTM layer with residual connection before output gate
    
    properties
        NumHiddenUnits
        OutputMode
        ProjectionMatrixType
    end
    
    properties (Learnable)
        InputWeights
        RecurrentWeights
        Bias
        ProjectionMatrix
    end
    
    properties (State)
        % Layer state parameters.
        HiddenState
        CellState
    end
    
    methods
        function layer = residualLSTMLayer(numHiddenUnits,args)
            %RESIDUALLSTMLAYER Residual layer
            %   layer = customLSTMLayer(numHiddenUnits, hasSameSizeInputOutput)
            %   creates a LSTM layer with the specified number of
            %   hidden units and projection matrix for residual connection
            %   if input and output have different size.
            %
            %   layer = customLSTMLayer(numHiddenUnits, hasSameSizeInputOutput,
            %                           Name=Value)
            %   creates a LSTM layer and specifies additional
            %   options using one or more name-value arguments:
            %
            %   Name       - Name of the layer, specified as a string.
            %                The default is "lstm".
            %
            %   OutputMode - Output mode, specified as one of the
            %                following:
            %                "sequence" - Output the entire sequence
            %                             of data.
            %
            %                 "last"     - Output the last time step
            %                              of the data.
            %                 The default is "sequence".
            %   ProjectionMatrixType - Type of projection matrix in
            %   Residual LSTM paper (Kim et al.) to be used:
            %                        - "fullyConnected" - uses
            %                        fullyConnectedLayer for projection
            %                        - "1dconv" - uses convolution1dLayer
            %                        for projection
            %                        The default is "1dconv"
            
            % Parse input arguments.
            arguments
                numHiddenUnits
                args.Name (1,1) string = "residual lstm";
                args.OutputMode (1,1) string {mustBeMember(args.OutputMode, {'last', 'sequence'})} = "sequence";
                args.ProjectionMatrixType (1,1) string {mustBeMember(args.ProjectionMatrixType, {'fullyConnected', '1dconv'})} = "1dconv";
            end
            
            layer.NumHiddenUnits = numHiddenUnits;
            layer.Name = args.Name;
            layer.OutputMode = args.OutputMode;
            layer.Description = "Residual LSTM with " + numHiddenUnits + " hidden units";
            layer.ProjectionMatrixType = args.ProjectionMatrixType;
            if strcmp(layer.ProjectionMatrixType, 'fullyConnected')
                lgraph = layerGraph(fullyConnectedLayer(numHiddenUnits));
            else % '1dconv'
                lgraph = layerGraph(convolution1dLayer(1, numHiddenUnits));
            end
            layer.ProjectionMatrix = dlnetwork(lgraph, initialize=false);
        end
        
        function layer = initialize(layer,layout)
            numHiddenUnits = layer.NumHiddenUnits;
            
            % Find number of channels.
            idx = finddim(layout,"C");
            numChannels = layout.Size(idx);
            
            % Initialize input weights.
            if isempty(layer.InputWeights)
                sz = [4*numHiddenUnits numChannels];
                numOut = 4*numHiddenUnits;
                numIn = numChannels;
                layer.InputWeights = layer.initializeGlorot(sz,numOut,numIn);
            end
            
            % Initialize recurrent weights.
            if isempty(layer.RecurrentWeights)
                sz = [4*numHiddenUnits numHiddenUnits];
                layer.RecurrentWeights = layer.initializeOrthogonal(sz);
            end
            
            % Initialize bias.
            if isempty(layer.Bias)
                layer.Bias = layer.initializeUnitForgetGate(numHiddenUnits);
            end
            
            % Initialize hidden state.
            if isempty(layer.HiddenState)
                layer.HiddenState = zeros(numHiddenUnits,1);
            end
            
            % Initialize cell state.
            if isempty(layer.CellState)
                layer.CellState = zeros(numHiddenUnits,1);
            end
            
            % Initialize projection matrix
            layer.ProjectionMatrix = initialize(layer.ProjectionMatrix, layout);
        end
        
        function [Z,cellState,hiddenState] = predict(layer,X)
            %PREDICT residual LSTM predict function
            %   [Z,hiddenState,cellState] = predict(layer,X) forward
            %   propagates the data X through the layer and returns the
            %   layer output Z and the updated hidden and cell states. X
            %   is a dlarray with format "CBT" and Z is a dlarray with
            %   format "CB" or "CBT", depending on the layer OutputMode
            %   property.
            
            % Initialize sequence output.
            numHiddenUnits = layer.NumHiddenUnits;
            miniBatchSize = size(X,finddim(X,"B"));
            numTimeSteps = size(X,finddim(X,"T"));
            
            if layer.OutputMode == "sequence"
                Z = zeros(numHiddenUnits,miniBatchSize,numTimeSteps,"like",X);
                Z = dlarray(Z,"CBT");
            end
            
            XStripped = stripdims(X);
            WX = pagemtimes(layer.InputWeights,XStripped);
            
            % Use only the projection matrix to simplify computation
            XPadded = predict(layer.ProjectionMatrix, X);
            XPadded = stripdims(XPadded);
            
            % %Old code with padding
            %if size(XStripped,1) <= numHiddenUnits
            %    % Implementation of residual block in tensorflow uses same size of padding both in front and back of the X.
            %    % (See implementation line 1593: https://github.com/tflearn/tflearn/blob/master/tflearn/layers/conv.py#L1593)
            %    padding_size = floor((numHiddenUnits-size(XStripped,1)) / 2);
            %    XPadded = zeros(numHiddenUnits,miniBatchSize,numTimeSteps,"like",X);
            %    XPadded(padding_size+1:end-padding_size,:,:) = XStripped;
            %else
            %    XPadded = predict(layer.ProjectionMatrix, X);
            %    XPadded = stripdims(XPadded);
            %    %disp(XPadded)
            %    %error("Input size %d must be less than or equal to the number of hidden units %d." + ...
            %    %    "Lower number of hidden units is not supported for residual LSTM.", size(X,1), numHiddenUnits);
            %end
            
            idx_i = 1:numHiddenUnits;
            idx_f = 1+numHiddenUnits:2*numHiddenUnits;
            idx_c = 1+2*numHiddenUnits:3*numHiddenUnits;
            idx_o = 1+3*numHiddenUnits:4*numHiddenUnits;
            
            hiddenState = layer.HiddenState;
            cellState = layer.CellState;
            
            for t = 1:numTimeSteps
                WH = layer.RecurrentWeights * hiddenState;
                
                % Equations from paper "Residual LSTM: Design of a Deep Recurrent Architecture for Distant Speech Recognition"
                it = ...
                    sigmoid(WX(idx_i,:,t) + WH(idx_i,:) + layer.Bias(idx_i,:)); % Equation (4)
                ft = ...
                    sigmoid(WX(idx_f,:,t) + WH(idx_f,:) + layer.Bias(idx_f,:)); % Equation (5)
                cellState = ...
                    ft .* cellState + ...
                    it .* tanh(WX(idx_c,:,t) + WH(idx_c,:) + layer.Bias(idx_c,:)); % Equation (6)
                ot = sigmoid(WX(idx_o,:,t) + WH(idx_o,:) + layer.Bias(idx_o,:)); % Equation (7)
                
                rt = tanh(cellState); % Equation (12)
                % We don't use the Equation (13) (projection/padding of rt) as the cell state will always have the same size as the hidden state.
                hiddenState = ot .* (rt + XPadded(:,:,t)); % Equation (15)
                
                % Update sequence output.
                if layer.OutputMode == "sequence"
                    Z(:,:,t) = hiddenState;
                end
            end
            
            % Last time step output.
            if layer.OutputMode == "last"
                Z = dlarray(hiddenState,"CB");
            end
            
        end
        
        function layer = resetState(layer)
            numHiddenUnits = layer.NumHiddenUnits;
            layer.HiddenState = zeros(numHiddenUnits,1);
            layer.CellState = zeros(numHiddenUnits,1);
        end
    end
    
    methods (Access = private)
        
        % Copy of matlab methods from https://www.mathworks.com/help/deeplearning/ug/initialize-learnable-parameters-for-custom-training-loop.html#mw_975f9595-ce05-4163-a5d7-1719d1609e0a
        % to avoid problems with undefined initialize functions in the initialize
        % method of the custom layer class.
        % accessed on 2023-05-27
        
        function weights = initializeGlorot(layer, sz,numOut,numIn)
            
            Z = 2*rand(sz,'single') - 1;
            bound = sqrt(6 / (numIn + numOut));
            
            weights = bound * Z;
            weights = dlarray(weights);
            
        end
        
        
        function parameter = initializeOrthogonal(layer, sz)
            
            Z = randn(sz,'single');
            [Q,R] = qr(Z,0);
            
            D = diag(R);
            Q = Q * diag(D ./ abs(D));
            
            parameter = dlarray(Q);
            
        end
        
        function bias = initializeUnitForgetGate(layer, numHiddenUnits)
            
            bias = zeros(4*numHiddenUnits,1,'single');
            
            idx = numHiddenUnits+1:2*numHiddenUnits;
            bias(idx) = 1;
            
            bias = dlarray(bias);
            
        end
        
    end
end

