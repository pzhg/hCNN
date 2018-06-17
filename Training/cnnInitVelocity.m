function [dW, dB]=cnnInitVelocity(cnn)

dW=cell(1, cnn.LNum);
dB=cell(1, cnn.LNum);
for iLayer=1:cnn.LNum
    switch cnn.Layers{iLayer}.type
        case {2, 3}
            dW{iLayer}=gpuArray.zeros(size(cnn.Layers{iLayer}.W));
            dB{iLayer}=gpuArray.zeros(size(cnn.Layers{iLayer}.B));
        case 1
            dW{iLayer}=struct;
            dW{iLayer}.Ka=0;
            dW{iLayer}.Kr=0;
    end
end