function cnn = cnnAddBNLayer(cnn, varargin)
    % Batched Normalization Layer
    % Mode: 1 for conv layer, 2 for FC layer

    BLayer = struct;
    BLayer.type = 11;
    BLayer.eps = 1e-6;
    BLayer.OutDim = cnn.Layers{cnn.LNum}.OutDim;

    switch cnn.Layers{cnn.LNum}.type
        case {1, 2}
            BLayer.mode = 1;
        case {3, 12}
            BLayer.mode = 2;
        otherwise
            BLayer.mode = varargin{1};
    end

    % BLayer.mode = mode;
    BLayer.FNum = cnn.Layers{cnn.LNum}.FNum;

    if BLayer.mode == 1 && cnn.to.useGPU == 1
        % CNN
        BLayer.FDim = cnn.Layers{cnn.LNum}.FDim;
        BLayer.gamma = gpuArray.ones(1, 1, BLayer.FNum, 1, 'single');
        BLayer.beta = gpuArray.zeros(1, 1, BLayer.FNum, 1, 'single');
        BLayer.mean = gpuArray.zeros(1, 1, BLayer.FNum, 1, 'single');
        BLayer.var = gpuArray.zeros(1, 1, BLayer.FNum, 1, 'single');
    elseif BLayer.mode == 2 && cnn.to.useGPU == 1
        % FC
        BLayer.gamma = gpuArray.ones(BLayer.OutDim, 1, 'single');
        BLayer.beta = gpuArray.zeros(BLayer.OutDim, 1, 'single');
        BLayer.mean = gpuArray.zeros(BLayer.OutDim, 1, 'single');
        BLayer.var = gpuArray.zeros(BLayer.OutDim, 1, 'single');
    elseif BLayer.mode == 1 && cnn.to.useGPU == 0
        % CNN
        BLayer.FDim = cnn.Layers{cnn.LNum}.FDim;
        BLayer.gamma = ones(1, 1, BLayer.FNum, 1);
        BLayer.beta = zeros(1, 1, BLayer.FNum, 1);
        BLayer.mean = zeros(1, 1, BLayer.FNum, 1);
        BLayer.var = zeros(1, 1, BLayer.FNum, 1);
    elseif BLayer.mode == 2 && cnn.to.useGPU == 0
        % FC
        BLayer.gamma = ones(BLayer.OutDim, 1);
        BLayer.beta = zeros(BLayer.OutDim, 1);
        BLayer.mean = zeros(BLayer.OutDim, 1);
        BLayer.var = zeros(BLayer.OutDim, 1);
    end

    BLayer.mom = single(0.9);
    cnn.LNum = cnn.LNum + 1;
    cnn.Layers{cnn.LNum} = BLayer;

end