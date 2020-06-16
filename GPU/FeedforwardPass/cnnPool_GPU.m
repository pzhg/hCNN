function [PLayer, pooledFeatures]=cnnPool_GPU(PLayer, convolvedFeatures)

numImages=size(convolvedFeatures, 4);
numFilters=PLayer.FNum;

pooledFeatures=zeros(PLayer.OutDim(1), PLayer.OutDim(2), PLayer.FNum, numImages, 'single');

switch PLayer.poolMethod
    case 'mean'
        parfor imageNum=1:numImages
            for featureNum=1:numFilters
                featuremap=convolvedFeatures(:, :, featureNum, imageNum);
                pooledFeaturemap=conv2(gpuArray(featuremap), gpuArray.ones(PLayer.poolDim(1), PLayer.poolDim(2), 'single')/(PLayer.poolDim(1)*PLayer.poolDim(2)), 'valid');
                pooledFeatures(:, :, featureNum, imageNum)=gather(pooledFeaturemap(1:PLayer.poolDim(1):end, 1:PLayer.poolDim(2):end));
                % clear pooledFeaturemap;
            end
        end
    case 'max'
        poolLocation=ones(1, PLayer.OutDim(1)*PLayer.OutDim(2), PLayer.FNum, numImages);
        parfor imageNum=1:numImages
            for featureNum=1:numFilters
                temp=im2col(convolvedFeatures(:, :, featureNum, imageNum), [PLayer.poolDim(1), PLayer.poolDim(2)], 'distinct');
                [m, i]=max(temp, [], 1);
                poolLocation(1, :, featureNum, imageNum)=i;
                pooledFeatures(:, :, featureNum, imageNum)=reshape(m, PLayer.OutDim(1), PLayer.OutDim(2)); 
                % clear temp, m;
            end
        end    
        PLayer.poolLocation=poolLocation;
end