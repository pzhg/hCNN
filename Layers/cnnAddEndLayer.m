function cnn = cnnAddEndLayer(cnn)
    % End layer for BLOB layers

    ELayer = struct;
    ELayer.type = 103;
    ELayer.OutDim = cnn.Layers{cnn.LNum}.OutDim;
    ELayer.FNum = cnn.Layers{cnn.LNum}.FNum;
    cnn.LNum = cnn.LNum + 1;
    cnn.Layers{cnn.LNum} = ELayer;

end