clc;
clear all;
close all;
reset(gpuDevice(1));    % Initialize GPU
dbstop if error

%% Load Data (MNIST)
load MNIST.mat;

%% Training Options
to.epochs=3;            % Epoch number
to.batch=400;           % Batch number
to.batch_size=150;      % Batch size
to.alpha=0.1;           % Learning rate
to.momentum=0.9;        % Momentum
to.mom=0.5;             % Initial momentum
to.momIncrease=20;      % Momemtum change iteration count
to.lambda=0.0001;       % Weight decay parameter (a.k.a. L2 regularization parameter)

%% Initialize CNN
cnn=cnnInit;

%% Configure Layers
cnn=cnnAddInputLayer(cnn, [28, 28], 1);
cnn=cnnAddConvLayer(cnn, [5, 5], 8, 'r');
cnn=cnnAddActivationLayer(cnn, 'ReLu');
cnn=cnnAddPoolLayer(cnn, 'max', [2, 2]);
cnn=cnnAddConvLayer(cnn, [5, 5], 8, 'r');
cnn=cnnAddActivationLayer(cnn, 'ReLu');
cnn=cnnAddPoolLayer(cnn, 'max', [2, 2]);
cnn=cnnAddReshapeLayer(cnn);
cnn=cnnAddFCLayer(cnn, 10, 'r');
cnn=cnnAddSoftMaxLayer(cnn);

%% Train CNN
[ERR, cnn]=cnnTrainBP(cnn, TrainData, LabelData, to);
figure;
plot(ERR(1, :));
figure;
plot(ERR(2, :));

%% Test CNN
acc=cnnTestData(cnn, VData, VLabel, 1000);
fprintf('Validation accuracy is: %f\n', acc);