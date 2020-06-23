function cnn = cnnAddOutputLayer(cnn, OutType)
    % Output Layer
    % Type:
    %   'softmax' : SoftMax (Cross Entropy)
    %   'mse' : MSE
    %   'end_BLOB' : Ending layer for BLOB layer

    ELayer = struct;
    
    ELayer.OutDim = cnn.Layers{cnn.LNum}.OutDim;
    
    switch OutType
        case {'softmax', 'Softmax', 'SOFTMAX', 'SoftMax'}
            ELayer.FNum = 1;
            ELayer.type = 4;
        case {'mse', 'MSE'}
            ELayer.FNum = 1;
            ELayer.type = 8;
        case {'end_BLOB', 'END_BLOB', 'end_blob'}
            ELayer.FNum = cnn.Layers{cnn.LNum}.FNum;
            ELayer.type = 103;
        otherwise
            error('Illegal Output Layer Type!');
    end

    ELayer.OutType = OutType;
    cnn.LNum = cnn.LNum + 1;
    cnn.Layers{cnn.LNum} = ELayer;

end