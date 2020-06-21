function cnn = cnnAddDropOutLayer(cnn, rate)
    % Dropout Layer

    DLayer = struct;
    DLayer.type = 12;

    if cnn.to.useGPU == 1
        DLayer.rate = gpuArray(single(rate));
    else
        DLayer.rate = rate;
    end

    DLayer.test = cnn.to.test;
    DLayer.OutDim = cnn.Layers{cnn.LNum}.OutDim;
    DLayer.FNum = cnn.Layers{cnn.LNum}.FNum;
    DLayer.Location = [];
    cnn.LNum = cnn.LNum + 1;
    cnn.Layers{cnn.LNum} = DLayer;
end
