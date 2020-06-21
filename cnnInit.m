function cnn = cnnInit(to)

    cnn = struct;
    cnn.LNum = 0;
    cnn.wCost = 0;
    cnn.Layers = {};
    cnn.OutData = {};
    cnn.Delta = {};
    cnn.dW = {};
    cnn.dB = {};
    cnn.W_grad = {};
    cnn.B_grad = {};
    cnn.to = to;

end