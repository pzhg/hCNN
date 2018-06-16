function [cnn, OutData]=cnnFeedForward(cnn, images)

numImages=size(images, 4);
OutData=cell(1, cnn.LNum);
cnn.wCost=0;
for iLayer=1:cnn.LNum
    switch cnn.Layers{iLayer}.type
        case 0
            % Input Layer
            OutData{iLayer}=gpuArray(images);
        case 1
            % Hybrid Convolution Layer
            OutData{iLayer}=cnnConvolveRadar(cnn.Layers{iLayer}, OutData{iLayer-1});
        case 2
            % Convolution Layers
            cnn.wCost=cnn.wCost+sum(cnn.Layers{iLayer}.W(:).^2);
            OutData{iLayer}=cnnConvolve(cnn.Layers{iLayer}, OutData{iLayer-1});
        case 3
            % Fully Connected Layers
            cnn.wCost=cnn.wCost+sum(cnn.Layers{iLayer}.W(:).^2);
            OutData{iLayer}=cnnFullConnected(cnn.Layers{iLayer}, OutData{iLayer-1});
        case 4
            % Softmax Layer
            OutData{iLayer}=cnnSoftMax(OutData{iLayer-1});
        case 5
            % Pooling Layers
            [cnn.Layers{iLayer}, OutData{iLayer}]=cnnPool(cnn.Layers{iLayer}, OutData{iLayer-1});
        case 6
            % Reshape Layer
            OutData{iLayer}=reshape(OutData{iLayer-1}, [], numImages);
        case 7
            % Activation Function Layer
            OutData{iLayer}=cnnActivate(cnn.Layers{iLayer}, OutData{iLayer-1});
    end
end