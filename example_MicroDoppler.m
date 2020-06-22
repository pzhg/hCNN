clc;
clear all;
close all;
dbstop if error

mpiprofile on;

%% Load Data (MicroDoppler)
load('MicroDoppler.mat');
% The MicroDoppler.mat file can be downloaded from https://1drv.ms/u/s!Akr-loZjbPYVufFK8J6pAMtHi1fEyA?e=T6Vgss
trainLabelNoCarRe = trainLabelNoCarRe + 1;
testLabelNoCarRe = testLabelNoCarRe + 1;

%% Training Options
to.epochs = 5;              % Epoch number
to.batch = 78;              % Batch number
to.batch_size = 64;         % Batch size
to.alpha = 0.1;             % Learning raten
to.momentum = 0.9;          % Momentum
to.mom = 0.5;               % Initial momentum
to.momIncrease = 20;        % Momemtum change iteration count
to.lambda = 0.0001;         % Weight decay parameter (a.k.a. L2 regularization parameter)
to.useGPU = 1;              % Use GPU

if to.useGPU == 1
    reset(gpuDevice(1));    % Initialize GPU
end

to.test = 0;

%% Initialize CNN
cnn = cnnInit(to);
feature('SetPrecision', 24);

%% Configure Layers
cnn = cnnAddInputLayer(cnn, [400, 144], 1);

    % subnet 1: CNN
    cnn1 = cnnInit(to);
    cnn1 = cnnAddInputLayer(cnn1, [400, 144], 1);
    cnn1 = cnnAddTransformLayer(cnn1, 'PCA', 1:8);
    cnn1 = cnnAddConvLayer(cnn1, [5, 5], 8, 'r');
    cnn1 = cnnAddBNLayer(cnn1);
    cnn1 = cnnAddActivationLayer(cnn1, 'ReLu');
    cnn1 = cnnAddPoolLayer(cnn1, 'max', [2, 2]);
    cnn1 = cnnAddConvLayer(cnn1, [3, 3], 8, 'r');
    cnn1 = cnnAddBNLayer(cnn1);
    cnn1 = cnnAddActivationLayer(cnn1, 'ReLu');
    cnn1 = cnnAddPoolLayer(cnn1, 'max', [2, 2]);
    cnn1 = cnnAddConvLayer(cnn1, [3, 3], 8, 'r');
    cnn1 = cnnAddBNLayer(cnn1);
    cnn1 = cnnAddActivationLayer(cnn1, 'ReLu');
    cnn1 = cnnAddPoolLayer(cnn1, 'max', [2, 2]);
    cnn1 = cnnAddConvLayer(cnn1, [3, 3], 8, 'r');
    cnn1 = cnnAddBNLayer(cnn1);
    cnn1 = cnnAddActivationLayer(cnn1, 'ReLu');
    cnn1 = cnnAddPoolLayer(cnn1, 'max', [2, 2]);
    cnn1 = cnnAddReshapeLayer(cnn1);
    cnn1 = cnnAddFCLayer(cnn1, 64, 'r');
    cnn1 = cnnAddActivationLayer(cnn1, 'ReLu');
    cnn1 = cnnAddEndLayer(cnn1);
    cnn1 = cnnInitVelocity(cnn1);

    % subnet 2: PCA
    cnn2 = cnnInit(to);
    cnn2 = cnnAddInputLayer(cnn2, [400, 144], 1);
    cnn2 = cnnAddTransformLayer(cnn2, 'PCA', 8:25);
    cnn2 = cnnAddConvLayer(cnn2, [5, 5], 8, 'r');
    cnn2 = cnnAddBNLayer(cnn2);
    cnn2 = cnnAddActivationLayer(cnn2, 'ReLu');
    cnn2 = cnnAddPoolLayer(cnn2, 'max', [2, 2]);
    cnn2 = cnnAddConvLayer(cnn2, [3, 3], 8, 'r');
    cnn2 = cnnAddBNLayer(cnn2);
    cnn2 = cnnAddActivationLayer(cnn2, 'ReLu');
    cnn2 = cnnAddPoolLayer(cnn2, 'max', [2, 2]);
    cnn2 = cnnAddConvLayer(cnn2, [3, 3], 8, 'r');
    cnn2 = cnnAddBNLayer(cnn2);
    cnn2 = cnnAddActivationLayer(cnn2, 'ReLu');
    cnn2 = cnnAddPoolLayer(cnn2, 'max', [2, 2]);
    cnn2 = cnnAddConvLayer(cnn2, [3, 3], 8, 'r');
    cnn2 = cnnAddBNLayer(cnn2);
    cnn2 = cnnAddActivationLayer(cnn2, 'ReLu');
    cnn2 = cnnAddPoolLayer(cnn2, 'max', [2, 2]);
    cnn2 = cnnAddReshapeLayer(cnn2);
    cnn2 = cnnAddFCLayer(cnn2, 64, 'r');
    cnn2 = cnnAddActivationLayer(cnn2, 'ReLu');
    cnn2 = cnnAddEndLayer(cnn2);
    cnn2 = cnnInitVelocity(cnn2);

% Combine
nets = {cnn1, cnn2};
cnn = cnnAddBLOBLayer(cnn, nets, 2, 128, 2);
cnn = cnnAddFCLayer(cnn, 128, 'r');
cnn = cnnAddBNLayer(cnn);
cnn = cnnAddActivationLayer(cnn, 'ReLu');
cnn = cnnAddFCLayer(cnn, 64, 'r');
cnn = cnnAddBNLayer(cnn);
cnn = cnnAddActivationLayer(cnn, 'ReLu');
cnn = cnnAddFCLayer(cnn, 2, 'r');
cnn = cnnAddSoftMaxLayer(cnn);
cnn = cnnInitVelocity(cnn);

%% Train CNN
[ERR, cnn] = cnnTrainBP(cnn, trainDataNoCarRe, trainLabelNoCarRe);
figure;
plot(ERR(1, :));
figure;
plot(ERR(2, :));

%% Test CNN
cnn.to.test = 1;
[acc, e] = cnnTestData(cnn, testDataNoCarRe, testLabelNoCarRe, 75);
fprintf('Validation accuracy is: %f\n', acc);
mpiprofile viewer;