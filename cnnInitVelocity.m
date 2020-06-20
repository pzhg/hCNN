function cnn=cnnInitVelocity(cnn)

cnn.dW=cell(1, cnn.LNum);
cnn.dB=cell(1, cnn.LNum);
for iLayer=1:cnn.LNum
    switch cnn.Layers{iLayer}.type
        case {2, 3}
            if cnn.to.useGPU==0
                cnn.dW{iLayer}=gpuArray.zeros(size(cnn.Layers{iLayer}.W));
                cnn.dB{iLayer}=gpuArray.zeros(size(cnn.Layers{iLayer}.B));
            else
                cnn.dW{iLayer}=gpuArray.zeros(size(cnn.Layers{iLayer}.W), 'single');
                cnn.dB{iLayer}=gpuArray.zeros(size(cnn.Layers{iLayer}.B), 'single');
            end
        case 1
            cnn.dW{iLayer}=struct;
            cnn.dW{iLayer}.Ka=0;
            cnn.dW{iLayer}.Kr=0;
        case 11
            cnn.dW{iLayer}.dgamma=0;
            cnn.dW{iLayer}.dbeta=0;
    end
end