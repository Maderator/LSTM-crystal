% THIS WAS NOT IMPLEMENTED BY US, BUT WAS USED IN THE COMPARISON OF THE MATLAB AND TENSORFLOW
% Copy of class defined in Define Custom Recurrent Deep Learning Layer (Deep Learning Toolbox) tutorial for Matlab version 2023a
% Link: https://www.mathworks.com/help/deeplearning/ug/define-custom-recurrent-deep-learning-layer.html
% Accessed on 25.05.2023

classdef peepholeLSTMLayer < nnet.layer.Layer & nnet.layer.Formattable
    %PEEPHOLELSTMLAYER Peephole LSTM Layer
    
    properties
        % Layer properties.
        
        NumHiddenUnits
        OutputMode
    end
    
    properties (Learnable)
        % Layer learnable parameters.
        
        InputWeights
        RecurrentWeights
        PeepholeWeights
        Bias
    end
    
    properties (State)
        % Layer state parameters.
        
        HiddenState
        CellState
    end
    
    methods
        function layer = peepholeLSTMLayer(numHiddenUnits,args)
            %PEEPHOLELSTMLAYER Peephole LSTM Layer
            %   layer = peepholeLSTMLayer(numHiddenUnits)
            %   creates a peephole LSTM layer with the specified number of
            %   hidden units.
            %
            %   layer = peepholeLSTMLayer(numHiddenUnits,Name=Value)
            %   creates a peephole LSTM layer and specifies additional
            %   options using one or more name-value arguments:
            %
            %      Name       - Name of the layer, specified as a string.
            %                   The default is "".
            %
            %      OutputMode - Output mode, specified as one of the
            %                   following:
            %                      "sequence" - Output the entire sequence
            %                                   of data.
            %
            %                      "last"     - Output the last time step
            %                                   of the data.
            %                   The default is "sequence".
            
            % Parse input arguments.
            arguments
                numHiddenUnits
                args.Name = "";
                args.OutputMode = "sequence";
            end
            
            layer.NumHiddenUnits = numHiddenUnits;
            layer.Name = args.Name;
            layer.OutputMode = args.OutputMode;
            
            % Set layer description.
            layer.Description = "Peephole LSTM with " + numHiddenUnits + " hidden units";
        end
        
        function layer = initialize(layer,layout)
            % layer = initialize(layer,layout) initializes the layer
            % learnable and state parameters.
            %
            % Inputs:
            %         layer  - Layer to initialize.
            %         layout - Data layout, specified as a
            %                  networkDataLayout object.
            %
            % Outputs:
            %         layer - Initialized layer.
            
            numHiddenUnits = layer.NumHiddenUnits;
            
            % Find number of channels.
            idx = finddim(layout,"C");
            numChannels = layout.Size(idx);
            
            % Initialize input weights.
            if isempty(layer.InputWeights)
                sz = [4*numHiddenUnits numChannels];
                numOut = 4*numHiddenUnits;
                numIn = numChannels;
                layer.InputWeights = initializeGlorot(sz,numOut,numIn);
            end
            
            % Initialize recurrent weights.
            if isempty(layer.RecurrentWeights)
                sz = [4*numHiddenUnits numHiddenUnits];
                layer.RecurrentWeights = initializeOrthogonal(sz);
            end
            
            % Initialize peephole weights.
            if isempty(layer.PeepholeWeights)
                sz = [3*numHiddenUnits 1];
                numOut = 3*numHiddenUnits;
                numIn = 1;
                
                layer.PeepholeWeights = initializeGlorot(sz,numOut,numIn);
            end
            
            % Initialize bias.
            if isempty(layer.Bias)
                layer.Bias = initializeUnitForgetGate(numHiddenUnits);
            end
            
            % Initialize hidden state.
            if isempty(layer.HiddenState)
                layer.HiddenState = zeros(numHiddenUnits,1);
            end
            
            % Initialize cell state.
            if isempty(layer.CellState)
                layer.CellState = zeros(numHiddenUnits,1);
            end
        end
        
        function [Z,cellState,hiddenState] = predict(layer,X)
            %PREDICT Peephole LSTM predict function
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
            
            % Calculate WX + b.
            X = stripdims(X);
            WX = pagemtimes(layer.InputWeights,X) + layer.Bias;
            
            % Indices of concatenated weight arrays.
            idx1 = 1:numHiddenUnits;
            idx2 = 1+numHiddenUnits:2*numHiddenUnits;
            idx3 = 1+2*numHiddenUnits:3*numHiddenUnits;
            idx4 = 1+3*numHiddenUnits:4*numHiddenUnits;
            
            % Initial states.
            hiddenState = layer.HiddenState;
            cellState = layer.CellState;
            
            % Loop over time steps.
            for t = 1:numTimeSteps
                % Calculate R*h_{t-1}.
                Rht = layer.RecurrentWeights * hiddenState;
                
                % Calculate p*c_{t-1}.
                pict = layer.PeepholeWeights(idx1) .* cellState;
                pfct = layer.PeepholeWeights(idx2) .* cellState;
                
                % Gate calculations.
                it = sigmoid(WX(idx1,:,t) + Rht(idx1,:) + pict);
                ft = sigmoid(WX(idx2,:,t) + Rht(idx2,:) + pfct);
                gt = tanh(WX(idx3,:,t) + Rht(idx3,:));
                
                % Calculate ot using updated cell state.
                cellState = gt .* it + cellState .* ft;
                poct = layer.PeepholeWeights(idx3) .* cellState;
                ot = sigmoid(WX(idx4,:,t) + Rht(idx4,:) + poct);
                
                % Update hidden state.
                hiddenState = tanh(cellState) .* ot;
                
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
            %RESETSTATE Reset layer state
            % layer = resetState(layer) resets the state properties of the
            % layer.
            
            numHiddenUnits = layer.NumHiddenUnits;
            layer.HiddenState = zeros(numHiddenUnits,1);
            layer.CellState = zeros(numHiddenUnits,1);
        end
    end
end