function cnn=cnnAddReshapeLayer(cnn)
% Reshape Layer from Convolutional to Fully Connected

RLayer=struct;
RLayer.type=6;
RLayer.OutDim=cnn.Layers{cnn.LNum}.OutDim(1)*cnn.Layers{cnn.LNum}.OutDim(2)*cnn.Layers{cnn.LNum}.FNum;
RLayer.FNum=1;
cnn.LNum=cnn.LNum+1;
cnn.Layers{cnn.LNum}=RLayer;