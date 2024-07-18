function valError = compareRNNs_depth()

RNNTypes = {'LSTM', 'GRU', 'peepholeLSTM', 'residualLSTM'};
depth = [1, 2, 4];
results = cell(length(RNNTypes)*length(depth), 3);

for m=1:length(depth)
    for n=1:length(RNNTypes)
        RNNType = RNNTypes(n);
        [valError, time] = trainMyRNNNetwork( ...
            RNNType, ...
            "datasetPath", "trainingShuffled.mat", ...
            "NetworkDepth", depth(m), ...
            "NumberOfEpochs",10, ...
            "MiniBatchSize", 32, ...
            "Repetitions", 4, ...
            "crossValidationFolds",1, ...
            "NumHiddenUnits",20);
        result = {char(RNNType + "_" + depth(m) + "_layers"), valError, time};
        results(n+m, :) = result;
        save("results_" + depth(m) + "_depth_" + RNNType + ".mat", 'result');
        disp(result);
    end
end
    fieldNames = {'model_name', 'RMSE', 'time'};
    results = cell2struct(results, fieldNames, 2);
    save("results_depth.mat", 'results');
end