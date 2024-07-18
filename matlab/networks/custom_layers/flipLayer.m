classdef flipLayer < nnet.layer.Layer ...
                & nnet.layer.Formattable
    %flipLayer flip the input data along the third dimension. Used for bidirectional RNNs construction.
        methods
                function layer = flipLayer(name)
                        layer.Name = name;
                end
                function Z = predict(layer, X)
                        Z = flip(X, 3);
                end
        end
end