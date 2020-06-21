function cnn = cnnAddActivationLayer(cnn, ActFunc)
    % Activation Function Layer
    %   ActFunc: 'ReLu'    -- ReLu function
    %            'LReLu'   -- Leaky ReLu function
    %            'ABS'     -- Absolute value function
    %            'Sigmoid' -- Sigmoid function

    RLayer = struct;
    RLayer.type = 7;
    RLayer.Leak = 0.02; % Leak Rate, adjustable
    RLayer.ActFunc = ActFunc;
    RLayer.OutDim = cnn.Layers{cnn.LNum}.OutDim;
    RLayer.FNum = cnn.Layers{cnn.LNum}.FNum;
    cnn.LNum = cnn.LNum + 1;
    cnn.Layers{cnn.LNum} = RLayer;

end