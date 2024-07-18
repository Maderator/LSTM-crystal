function rnnLayerClass = chooseRNNLayerClass(rnnType)
arguments
    rnnType char {mustBeMember(rnnType, {'GRU', 'LSTM', 'peepholeLSTM', 'residualLSTM'})} = 'residualLSTM'
end

switch rnnType
    case "LSTM"
        rnnLayerClass = @lstmLayer;
    case "GRU"
        rnnLayerClass = @gruLayer;
    case "peepholeLSTM"
        rnnLayerClass = @peepholeLSTMLayer;
    case "residualLSTM"
        rnnLayerClass = @residualLSTMLayer;
end
end