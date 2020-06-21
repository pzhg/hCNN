function cnn = cnnAddCoPCALayer(cnn, FDim, PCADim, PCAStep, CorrType)
    % Corrtype: 1 for Auto-correlation, 2 for Cross-correlation

    CLayer = struct;
    CLayer.type = 102;
    CLayer.FDim = FDim;
    CLayer.PCADim = PCADim;
    CLayer.PCAStep = PCAStep;
    CLayer.CorrType = CorrType;
    % CLayer.calced=0;
    % CLayer.useGPU=cnn.to.useGPU;
    switch CLayer.CorrType
        case 1
            CLayer.FNum = cnn.Layers{cnn.LNum}.FNum;
        case 2
            CLayer.FNum = cnn.Layers{cnn.LNum}.FNum * (cnn.Layers{cnn.LNum}.FNum - 1) / 2;
        otherwise
            error('Correlation Type Error!');
    end

    Dim = ((cnn.Layers{cnn.LNum}.OutDim(1) - CLayer.FDim(1)) / CLayer.PCAStep(1) + 1) * ((cnn.Layers{cnn.LNum}.OutDim(2) - CLayer.FDim(2)) / CLayer.PCAStep(2) + 1);
    CLayer.OutDim = [Dim, Dim];
    cnn.LNum = cnn.LNum + 1;
    cnn.Layers{cnn.LNum} = CLayer;

end