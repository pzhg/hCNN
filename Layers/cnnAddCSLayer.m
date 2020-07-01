function cnn = cnnAddCSLayer(cnn, OutDim, comp)

    CLayer = struct;
    CLayer.type = 101;
    CLayer.calced = 0;
    CLayer.FNum = cnn.Layers{cnn.LNum}.FNum;
    % CLayer.FDim=OutDim;
    CLayer.OutDim = [OutDim, 1];
    % CLayer.useGPU=cnn.to.useGPU;
    if cnn.to.useGPU == 1
        CLayer.A = gpuArray.randn(CLayer.OutDim(1), cnn.Layers{cnn.LNum}.OutDim(1) * cnn.Layers{cnn.LNum}.OutDim(2), 'single');
    else
        CLayer.A = randn(CLayer.OutDim(1), cnn.Layers{cnn.LNum}.OutDim(1) * cnn.Layers{cnn.LNum}.OutDim(2));
    end
    
    if comp == 'c'
        
        if cnn.to.useGPU == 1
            CLayer.A = CLayer.A + 1j * gpuArray.randn(CLayer.OutDim(1), cnn.Layers{cnn.LNum}.OutDim(1) * cnn.Layers{cnn.LNum}.OutDim(2), 'single');
        else
            CLayer.A = CLayer.A + 1j *  randn(CLayer.OutDim(1), cnn.Layers{cnn.LNum}.OutDim(1) * cnn.Layers{cnn.LNum}.OutDim(2));
        end
    
    end

    cnn.LNum = cnn.LNum + 1;
    cnn.Layers{cnn.LNum} = CLayer;

end