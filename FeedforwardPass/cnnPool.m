function [PLayer, pooledFeatures]=cnnPool(PLayer, convolvedFeatures)

numImages=size(convolvedFeatures, 4);
numFilters=PLayer.FNum;

pooledFeatures=gpuArray.zeros(PLayer.OutDim(1), PLayer.OutDim(2), PLayer.FNum, numImages);
PLayer.poolLocation=gpuArray.ones(PLayer.OutDim(1)*PLayer.poolDim(1), PLayer.OutDim(2)*PLayer.poolDim(2), PLayer.FNum, numImages);

switch PLayer.poolMethod
    case 'mean'
        parfor imageNum=1:numImages
            for featureNum=1:numFilters
                featuremap=squeeze(convolvedFeatures(:, :, featureNum, imageNum));
                pooledFeaturemap=conv2(featuremap, gpuArray.ones(PLayer.poolDim)/(PLayer.poolDim(1)*PLayer.poolDim(2)), 'valid');
                pooledFeatures(:, :, featureNum, imageNum)=pooledFeaturemap(1:PLayer.poolDim(1):end, 1:PLayer.poolDim(2):end);
            end
        end
    case 'max'
        for imageNum=1:numImages
            for featureNum=1:numFilters
                temp=im2col(convolvedFeatures(:, :, featureNum, imageNum), PLayer.poolDim, 'distinct');
                [m, i]=max(temp);
                temp=zeros(size(temp));
                temp(sub2ind(size(temp), i, 1:size(i, 2)))=1;
                PLayer.poolLocation(:, :, featureNum, imageNum)=gpuArray(col2im(temp, PLayer.poolDim, PLayer.OutDim.*PLayer.poolDim, 'distinct'));
                pooledFeatures(:, :, featureNum, imageNum)=reshape(m, size(pooledFeatures, 1), size(pooledFeatures, 2)); 
            end
        end    
end