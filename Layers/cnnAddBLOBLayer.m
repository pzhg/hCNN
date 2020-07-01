function cnn = cnnAddBLOBLayer(cnn, Nets, OutDim, combineType)
    % Nets: cells of subnets
    % NNum: number of subnets
    % OutDim: dimension of output of this layer
    % combineType: 1 for addition, 2 for linking

    BLayer = struct;
    BLayer.type = 10;
    BLayer.Nets = Nets;
    BLayer.NNum = length(Nets) ;
    BLayer.OutDim = OutDim;
    % for inet=1:BLayer.NNum
    %     tcnn=BLayer.Nets{inet};
    %     BLayer.OutDim=BLayer.OutDim+tcnn.Layers{tcnn.LNum}.OutDim;
    % end
    BLayer.combineType = combineType;
    % BLayer.useGPU=cnn.to.useGPU;
    cnn.LNum = cnn.LNum + 1;
    cnn.Layers{cnn.LNum} = BLayer;

end