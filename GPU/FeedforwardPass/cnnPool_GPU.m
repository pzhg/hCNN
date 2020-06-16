function [PLayer, pooledFeatures]=cnnPool_GPU(PLayer, convolvedFeatures)

numImages=size(convolvedFeatures, 4);
numFilters=PLayer.FNum;

pooledFeatures=gpuArray.zeros(PLayer.OutDim(1), PLayer.OutDim(2), PLayer.FNum, numImages, 'single');
PLayer.poolLocation=gpuArray.ones(1, PLayer.OutDim(1)*PLayer.OutDim(2), PLayer.FNum, numImages, 'single');

switch PLayer.poolMethod
    case 'mean'
        parfor imageNum=1:numImages
            for featureNum=1:numFilters
                featuremap=squeeze(convolvedFeatures(:, :, featureNum, imageNum));
                pooledFeaturemap=conv2(featuremap, gpuArray.ones(PLayer.poolDim(1), PLayer.poolDim(2), 'single')/(PLayer.poolDim(1)*PLayer.poolDim(2)), 'valid');
                pooledFeatures(:, :, featureNum, imageNum)=pooledFeaturemap(1:PLayer.poolDim(1):end, 1:PLayer.poolDim(2):end);
                % clear pooledFeaturemap;
            end
        end
    case 'max'
        for imageNum=1:numImages
            for featureNum=1:numFilters
                temp=im2col(convolvedFeatures(:, :, featureNum, imageNum), [PLayer.poolDim(1), PLayer.poolDim(2)], 'distinct');
                [m, i]=max(temp, [], 1);
                PLayer.poolLocation(1, :, featureNum, imageNum)=i;
                pooledFeatures(:, :, featureNum, imageNum)=single(reshape(m, PLayer.OutDim(1), PLayer.OutDim(2))); 
                % clear temp, m;
            end
        end    
end