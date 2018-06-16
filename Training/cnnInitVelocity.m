function [dW, dB]=cnnInitVelocity(cnn)

dW=cell(1, cnn.LNum);
dB=cell(1, cnn.LNum);
for iLayer=1:cnn.LNum
    if cnn.Layers{iLayer}.type==2 || cnn.Layers{iLayer}.type==3
        dW{iLayer}=gpuArray.zeros(size(cnn.Layers{iLayer}.W));
        dB{iLayer}=gpuArray.zeros(size(cnn.Layers{iLayer}.B));
    end
end