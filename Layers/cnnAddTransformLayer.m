function cnn = cnnAddTransformLayer(cnn, TName, varargin)
    % Input Layer
    %   inputSize: dimensionality of input data, [x-dim, y-dim]
    %   channelNum: number of channels of input data

    CLayer = struct;
    CLayer.type = 104;
    CLayer.TName = TName;
    CLayer.FNum = cnn.Layers{cnn.LNum}.FNum;
    CLayer.OutDim = cnn.Layers{cnn.LNum}.OutDim;

    switch TName
        case {'PCA', 'pca'}
            CLayer.PCADim = varargin{1};
        case {'MAXPOOL', 'maxpool', 'MEANPOOL', 'meanpool'}
            CLayer.poolDim = varargin{1};
            CLayer.OutDim = CLayer.OutDim ./ CLayer.poolDim;
    end

    % CLayer.useGPU=cnn.to.useGPU;
    cnn.LNum = cnn.LNum + 1;
    cnn.Layers{cnn.LNum} = CLayer;

end
