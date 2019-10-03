function cnn=cnnInitVelocity(cnn)

cnn.dW=cell(1, cnn.LNum);
cnn.dB=cell(1, cnn.LNum);
for iLayer=1:cnn.LNum
    switch cnn.Layers{iLayer}.type
        case {2, 3}
            cnn.dW{iLayer}=single(gpuArray.zeros(size(cnn.Layers{iLayer}.W)));
            cnn.dB{iLayer}=single(gpuArray.zeros(size(cnn.Layers{iLayer}.B)));
        case 1
            cnn.dW{iLayer}=struct;
            cnn.dW{iLayer}.Ka=0;
            cnn.dW{iLayer}.Kr=0;
    end
end