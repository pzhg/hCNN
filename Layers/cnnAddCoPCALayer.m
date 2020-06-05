function cnn=cnnAddCoPCALayer(cnn, FDim, PCADim, PCAStep, CorrType)

CLayer=struct;
CLayer.type=102;
CLayer.FDim=FDim;
CLayer.PCADim=PCADim;
CLayer.PCAStep=PCAStep;
CLayer.CorrType=CorrType;
% CLayer.calced=0;
CLayer.useGPU=cnn.to.useGPU;
if CLayer.CorrType==1
    CLayer.FNum=cnn.Layers{cnn.LNum}.FNum;
else
    CLayer.FNum=cnn.Layers{cnn.LNum}.FNum*(cnn.Layers{cnn.LNum}.FNum-1)/2;
end
Dim=((cnn.Layers{cnn.LNum}.OutDim(1)-CLayer.FDim(1))/CLayer.PCAStep(1)+1)*((cnn.Layers{cnn.LNum}.OutDim(2)-CLayer.FDim(2))/CLayer.PCAStep(2)+1);
CLayer.OutDim=[Dim, Dim];
cnn.LNum=cnn.LNum+1;
cnn.Layers{cnn.LNum}=CLayer;