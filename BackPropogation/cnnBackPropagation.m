function Delta=cnnBackPropagation(cnn, OutData, outPut)

Delta=cell(1, cnn.LNum);
for iLayer=cnn.LNum:-1:1
    switch cnn.Layers{iLayer}.type
        case 4
            % Error of SoftMax Layer
            Delta{iLayer}=OutData{iLayer}-outPut;
        case 3
            % Error of Fully Connected Layer
            Delta{iLayer}=cnnDeFullConnected(cnn.Layers{iLayer}, Delta{iLayer+1});
        case 5
            % Error of Pooling Layer
            Delta{iLayer}=cnnDePool(cnn.Layers{iLayer}, Delta{iLayer+1});
        case 7
            % Error of Activation Layer
            Delta{iLayer}=cnnDeActivate(cnn.Layers{iLayer}, Delta{iLayer+1}, OutData{iLayer}, OutData{iLayer-1});
        case 2
            % Error of Convolution Layer
            Delta{iLayer}=cnnDeConv(cnn.Layers{iLayer}, Delta{iLayer+1});
        case 6
            % Error of Reshape Layer
            Delta{iLayer}=reshape(Delta{iLayer+1}, size(OutData{iLayer-1}));
        case 1
            % Error of Hybrid Convolution Layer
            [Ka, Kr]=cnnDeconvolveRadar(cnn.Layers{iLayer}, Delta{iLayer+1}, OutData{iLayer-1});
            Delta{iLayer}.Ka=Ka;
            Delta{iLayer}.Kr=Kr;
    end
end