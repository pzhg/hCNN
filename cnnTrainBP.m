function [ERR, cnn] = cnnTrainBP(cnn, X, Y)
    % Train CNN using BP gradient descend method
    % Input:
    %   X: training data, [x-dim, y-dim, channel-num, batch-size,
    %                                batch-count]
    %   Y: label data, [1, batch-size, batch-count]
    % Output:
    %   ERR: array contains the accuracy and cost in each iteration
    %   cnn: the trained CNN

    if cnn.to.useGPU == 1
        [ERR, cnn] = cnnTrainBP_GPU(cnn, X, Y);
    else
        [ERR, cnn] = cnnTrainBP_CPU(cnn, X, Y);
    end

end