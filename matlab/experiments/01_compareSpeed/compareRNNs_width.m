function valError = compareRNNs_width()

RNNTypes = {'LSTM', 'GRU', 'peepholeLSTM', 'residualLSTM'};
width = [10, 20, 50, 100, 200, 400];
results = cell(length(RNNTypes)*length(width), 3);

for m=1:length(width)
    for n=1:length(RNNTypes)
        RNNType = RNNTypes(n);
        [valError, time] = trainMyRNNNetwork( ...
            RNNType, ...
            "datasetPath", "trainingShuffled.mat", ...
            "NetworkDepth", 1, ...
            "NumberOfEpochs",10, ...
            "MiniBatchSize", 32, ...
            "Repetitions", 4, ...
            "crossValidationFolds",1, ...
            "NumHiddenUnits",width(m));
        result = {char(RNNType + "_" + width(m) + "_units"), valError, time};
        results(n+(m-1)*length(RNNTypes), :) = result;
        save("results_" + width(m) + "_width_" + RNNType + ".mat", 'result');
        disp(result);
    end
end
    fieldNames = {'model_name', 'RMSE', 'time'};
    results = cell2struct(results, fieldNames, 2);
    save("results_width.mat", 'results');
end
