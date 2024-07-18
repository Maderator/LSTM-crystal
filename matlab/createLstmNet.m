
function net = createLstmNet(varargin)
% CREATELSTMNET - creates a Long short-Term Memory network

% Input arguments:
opts.seed = 0;
opts.epochs=50;
opts.start_learning_rate=1e-3;
opts.end_learning_rate=1e-5;
opts.learning_rate_decay=1e-4;
opts.learning_rate_decay_period=10;
opts.learning_rate_decay_method='exponential';
opts.momentum=0.9;
opts.weight_decay=0.0005;
opts.batch_size=32;
opts.gpu = 0;
opts = vl_argparse(opts, varargin);


load recurrentCrystalGrowth
%%
%
%%
% To get some idea what the variables |Inputs| and Out|puts| contain, we show
% their first 10 rows of the 1st sample as a matrix.

reshape(Inputs(1,1:10,:),[10,2])

reshape(Outputs(1,1:10,:),[10,6])
%%
% First of all, we normaliize the data into the interval [0,1], separately for
% the fluxes, for the temperatures measured by sensors, and for the position of
% the solid-liquid interface.

MinFlux = min(min(min(Inputs)));

MinT = min(min(min(Outputs(:,:,1:5))));

MinGamma = min(min(Outputs(:,:,6)));

DeltaFlux = max(max(max(Inputs)))-MinFlux;

DeltaT = max(max(max(Outputs(:,:,1:5))))-MinT;

DeltaGamma = max(max(Outputs(:,:,6)))-MinGamma;

NormalizedInputs = (Inputs-MinFlux)/DeltaFlux;

NormalizedOutputs(:,:,1:5) = (Outputs(:,:,1:5)-MinT)/DeltaT;

NormalizedOutputs(:,:,6) = (Outputs(:,:,6)-MinGamma)/DeltaGamma;
%%
% For the first 10 rows of the 1st sample, we check that the data are now really
% in the interval [0,1].

reshape(NormalizedInputs(1,1:10,:),[10,2])

reshape(NormalizedOutputs(1,1:10,:),[10,6])
%%
% Differently to the multilayer perceptron example, only the approach without
% cross-validation will be considered here for an LSTM network. We use 10% of
% the 500 samples, i.e., 50 samples for validation and the remaing ones for training.
% To this end, we take a random permutation of the numbers 1,...,500 and assign
% its last 50 elements as indices to the validation samples and the remaining
% 450 elements as indices to the training samples.

Permutation = randperm(500);

WhichTrain = Permutation(1:450);

WhichValidation = Permutation(451:500);
%%
% Let us decide that the variable |Outputs| from the previous 3 time steps returns
% back to the network input neurons, where it is combined with the variable |Inputs|.
% Hence, the network inputs actually consist of 4 parts: the variable |Inputs|,
% and the variable |Outputs| from 1, 2 and 3 time steps earlier. Observe that
% they are available only from the 4th to the 100th time step.

for d=1:500
    
    Inputs1{d} = reshape(NormalizedInputs(d,4:100,:),97,2)';
    
    Inputs2{d} = reshape(NormalizedOutputs(d,3:99,:),97,6)';
    
    Inputs3{d} = reshape(NormalizedOutputs(d,2:98,:),97,6)';
    
    Inputs4{d} = reshape(NormalizedOutputs(d,1:97,:),97,6)';
    
end
%%
% Making use of the above variables WhichTrain and WhichValidation, we obtain
% the final training and validation data. Observe that also the output training
% and validation data are available only from the 4th to the 100th time step.

for d=1:length(WhichTrain)
    
    Index = WhichTrain(d);
    
    TrainingInputs{d}=[Inputs1{Index};Inputs2{Index};Inputs3{Index};Inputs4{Index}];
    
    TrainingOutputs{d}=reshape(NormalizedOutputs(Index,4:100,:),97,6)';
    
end

for d=1:length(WhichValidation)
    
    Index = WhichValidation(d);
    
    ValidationInputs{d}=[Inputs1{Index};Inputs2{Index};Inputs3{Index};Inputs4{Index}];
    
    ValidationOutputs{d}=reshape(NormalizedOutputs(Index,4:100,:),97,6)';
    
end
%%
% The most important property of each deep neural network is which layers it
% contains, and how they are organized. This is specified with either a vector
% or a graph of objects belonging to the |layer| class. In particular, a vector
% of layers  is a sequential structure, hence, the computation proceeds from the
% previous layer in the vector to the next one.

Layers = [sequenceInputLayer(20,'Name','Input'),...
    lstmLayer(100,'OutputMode','sequence','Name','LSTMLayer'),...
    fullyConnectedLayer(6,'Name','MLPLayer'),...
    regressionLayer('Name','RegressionOutput')];
%%
% We can view the structure of the network layers using the following commmand.

plot(layerGraph(Layers))
%%
% Needless to say, the sequential structure is very simple, thus viewing is
% hardly needed. The previous command, however, will plot any directed acyclic
% graph structure of a deep neural network.
%%
% Finally, options for network training can be set. Most important among them
% are selecting one of the available optimizers, the size of batch for training
% the network (aka minibatch) and the names of variables containing the data for
% network variables. For network training, we have chosen  the commonly used stochastic
% gradient optimizer called 'adam'.

Options = trainingOptions('adam','MiniBatchSize',225,'Shuffle','never','MaxEpochs',100,...
    'ValidationData',{ValidationInputs,ValidationOutputs},'ValidationFrequency',50,'VerboseFrequency',50);
%%
% With the function |train| that we learned earlier, only particular kinds of
% neural networks with layers of a single type can be trained, such as multilayer
% perceptrons. For other particular kinds of neural networks, with particular
% mixtures of layers of different types, the function |trainNetwork| is needed.

LSTM = trainNetwork(TrainingInputs,TrainingOutputs,Layers,Options);
%%
% Whereas in the case of neural networks trained with the fuction |train|, the
% value computed by the network for a new input can be obtained simply through
% calling the network as a function for the considered input, in the case of networks
% trained with |trainNetwork|, the computed value can't be obtained in such a
% straightforward way. Instead, it is necessary to call the function predict
% if the network performs regression, or classify if it performs classification.

Predictions = predict(LSTM,ValidationInputs);
%%
% Once we have obtained the predictions for  the validation data, we can calculate
% any kind  of error that the network makes on the validation data, e.g., the
% root mean squared error of each validation sequence.  Main characteristics of
% the distribution of those errors then can be visualized with a box plot.

for d=1:length(WhichValidation)
    
    RMSE(d) = sqrt(mean((Predictions{d}(:)-ValidationOutputs{d}(:)).^2));
    
end

boxplot(RMSE)
%%
% Observe that the predictions were obtained with the normalized data, therefore,
% they correspond to the range of the normalized outputs. If we want to work with
% them in the range of the original outputs, we need to unnormalize them first,
% inversely to the normalization of the outputs.

for d=1:length(WhichValidation)
    
    UnnormalizedPredictions{d}(:,1:5) = DeltaT*Predictions{d}(:,1:5)+MinT;
    
    UnnormalizedPredictions{d}(:,6) = DeltaGamma*Predictions{d}(:,6)+MinGamma;
    
end
end