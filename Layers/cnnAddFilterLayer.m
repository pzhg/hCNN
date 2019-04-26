function cnn=cnnAddFilterLayer(cnn, FDim, PCADim, PCAStep)
% Input Layer
%   inputSize: dimensionality of input data, [x-dim, y-dim]
%   channelNum: number of channels of input data

CLayer=struct;
CLayer.type=9;
CLayer.FDim=FDim;
% CLayer.FNum=cnn.Layers{cnn.LNum}.FNum;
CLayer.FltDim=((cnn.Layers{1}.OutDim(1)-FDim(1))/PCAStep(1)+1)*((cnn.Layers{1}.OutDim(2)-FDim(2))/PCAStep(2)+1);
CLayer.A=gpuArray.randn(CLayer.FltDim, cnn.Layers{1}.OutDim(1)*cnn.Layers{1}.OutDim(2));
CLayer.PCADim=PCADim;
CLayer.PCAStep=PCAStep;
CLayer.OutDim=cnn.Layers{cnn.LNum}.OutDim+CLayer.FltDim*(CLayer.FltDim+1);
% 
% for ichannel=1:CLayer.FNum
%     switch CLayer.FilterType{ichannel}
%         case 'pca'
%             CLayer.FNum=CLayer.FNum+CLayer.PCADim;
%         case 'cov'
%             CLayer.FNum=CLayer.FNum+1;
%         case 'cs'
%             CLayer.FNum=CLayer.FNum+1;
%             A=randn(Cu
cnn.LNum=cnn.LNum+1;
cnn.Layers{cnn.LNum}=CLayer;