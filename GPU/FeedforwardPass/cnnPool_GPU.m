function [PLayer, pooledFeatures] = cnnPool_GPU(PLayer, convolvedFeatures)

    numImages = size(convolvedFeatures, 4);
    numFilters = PLayer.FNum;

    switch PLayer.poolMethod
        case {'mean', 'MEAN', 'Mean'}
            % Mean Pooling
            pooledFeatures = gpuArray.zeros(PLayer.OutDim(1), PLayer.OutDim(2), PLayer.FNum, numImages, 'single');

            parfor imageNum = 1:numImages
                for featureNum = 1:numFilters
                    featuremap = convolvedFeatures(:, :, featureNum, imageNum);
                    pooledFeaturemap = conv2(featuremap, gpuArray.ones(PLayer.poolDim(1), PLayer.poolDim(2), 'single') / (PLayer.poolDim(1) * PLayer.poolDim(2)), 'valid');
                    pooledFeatures(:, :, featureNum, imageNum) = pooledFeaturemap(1:PLayer.poolDim(1):end, 1:PLayer.poolDim(2):end);
                    % clear pooledFeaturemap;
                end
            end

        case {'max', 'MAX', 'Max'}
            % Max Pooling
            poolLocation = zeros(1, PLayer.OutDim(1) * PLayer.OutDim(2), PLayer.FNum, numImages);
            convolvedFeatures_ = gather(convolvedFeatures);
            pooledFeatures_ = zeros(PLayer.OutDim(1), PLayer.OutDim(2), PLayer.FNum, numImages, 'single');

            parfor imageNum = 1:numImages
                for featureNum = 1:numFilters
                    temp = im2col(convolvedFeatures_(:, :, featureNum, imageNum), [PLayer.poolDim(1), PLayer.poolDim(2)], 'distinct');
                    [m, i] = max(temp, [], 1);
                    poolLocation(1, :, featureNum, imageNum) = i;
                    pooledFeatures_(:, :, featureNum, imageNum) = reshape(m, PLayer.OutDim(1), PLayer.OutDim(2));
                    % clear temp, m;
                end
            end

            PLayer.poolLocation = poolLocation;
            pooledFeatures = gpuArray(pooledFeatures_);

        otherwise
            error('Unknown Pooling Method!');
    end

end