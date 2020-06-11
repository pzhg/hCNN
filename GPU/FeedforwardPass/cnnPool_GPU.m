function [PLayer, pooledFeatures]=cnnPool(PLayer, convolvedFeatures)

numImages=size(convolvedFeatures, 4);
numFilters=PLayer.FNum;

if PLayer.useGPU==1
    pooledFeatures=single(gpuArray.zeros(PLayer.OutDim(1), PLayer.OutDim(2), PLayer.FNum, numImages));
    PLayer.poolLocation=single(gpuArray.ones(1, PLayer.OutDim(1)*PLayer.OutDim(2), PLayer.FNum, numImages));
else
    pooledFeatures=single(zeros(PLayer.OutDim(1), PLayer.OutDim(2), PLayer.FNum, numImages));
    PLayer.poolLocation=single(ones(1, PLayer.OutDim(1)*PLayer.OutDim(2), PLayer.FNum, numImages));
end

switch PLayer.poolMethod
    case 'mean'
        parfor imageNum=1:numImages
            for featureNum=1:numFilters
                featuremap=squeeze(convolvedFeatures(:, :, featureNum, imageNum));
                if PLayer.useGPU==1
                    pooledFeaturemap=single(conv2(featuremap, gpuArray.ones(PLayer.poolDim(1), PLayer.poolDim(2))/(PLayer.poolDim(1)*PLayer.poolDim(2)), 'valid'));
                else
                    pooledFeaturemap=single(conv2(featuremap, ones(PLayer.poolDim(1), PLayer.poolDim(2))/(PLayer.poolDim(1)*PLayer.poolDim(2)), 'valid'));
                end
                pooledFeatures(:, :, featureNum, imageNum)=pooledFeaturemap(1:PLayer.poolDim(1):end, 1:PLayer.poolDim(2):end);
                clear pooledFeaturemap;
            end
        end
    case 'max'
        for imageNum=1:numImages
            for featureNum=1:numFilters
                temp=im2col(convolvedFeatures(:, :, featureNum, imageNum), [PLayer.poolDim(1), PLayer.poolDim(2)], 'distinct');
                [m, i]=max(temp, [], 1);
                PLayer.poolLocation(1, :, featureNum, imageNum)=i;
                pooledFeatures(:, :, featureNum, imageNum)=reshape(m, PLayer.OutDim(1), PLayer.OutDim(2)); 
                clear temp, m;
            end
        end    
end