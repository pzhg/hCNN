clc;
clear;
close all;
dbstop if error

profile on;
tic;

%% Load Data (MNIST)
load MNIST.mat;
% TrainData=reshape(TrainData, 28, 28, 1, 10, 6000);
% LabelData=reshape(LabelData, 1, 10, 6000);
% TrainData=TrainData(:, :, :, :, 1:10);
% LabelData=LabelData(:, :, 1:10);

%% Training Options
to.epochs = 3;              % Epoch number
to.batch = 400;             % Batch number
to.batch_size = 150;        % Batch size
to.alpha = 0.1;             % Learning rate
to.momentum = 0.9;          % Momentum
to.mom = 0.5;               % Initial momentum
to.momIncrease = 20;        % Momemtum change iteration count
to.lambda = 0.0001;         % Weight decay parameter (a.k.a. L2 regularization parameter)
to.test = 0;                % Flag for training or test
to.useGPU = 1;              % Use GPU

if to.useGPU == 1
    reset(gpuDevice(1));    % Initialize GPU
end

%% Initialize NN
cnn = cnnInit(to);
feature('SetPrecision', 24);

%% Configure Layers
cnn = cnnAddInputLayer(cnn, [28, 28], 1);
cnn = cnnAddConvLayer(cnn, [3, 3], 8, 'r');
cnn = cnnAddBNLayer(cnn, 1);
cnn = cnnAddActivationLayer(cnn, 'ReLu');
cnn = cnnAddPoolLayer(cnn, 'max', [2, 2]);
cnn = cnnAddConvLayer(cnn, [4, 4], 8, 'r');
cnn = cnnAddBNLayer(cnn, 1);
cnn = cnnAddActivationLayer(cnn, 'ReLu');
cnn = cnnAddPoolLayer(cnn, 'max', [2, 2]);
cnn = cnnAddReshapeLayer(cnn);
cnn = cnnAddFCLayer(cnn, 128, 'r');
cnn = cnnAddDropOutLayer(cnn, 0.3);
cnn = cnnAddBNLayer(cnn, 2);
cnn = cnnAddActivationLayer(cnn, 'ReLu');
cnn = cnnAddFCLayer(cnn, 10, 'r');
% cnn=cnnAddBNLayer(cnn, 2);
cnn = cnnAddSoftMaxLayer(cnn);

%% Train CNN
cnn = cnnInitVelocity(cnn); % Initial the NN Parameters
[ERR, cnn] = cnnTrainBP(cnn, TrainData, LabelData);
% figure;
% plot(ERR(1, :));
% figure;
% plot(ERR(2, :));

%% Test NN
cnn.to.test = 1;            % Set the test flag to 1
acc = cnnTestData(cnn, VData, VLabel, 1000);
fprintf('Validation accuracy is: %f\n', acc);

toc;
profile viewer;