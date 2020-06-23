function [PLayer, pooledFeatures] = cnnPool(PLayer, convolvedFeatures)

    numImages = size(convolvedFeatures, 4);
    numFilters = PLayer.FNum;

    pooledFeatures = zeros(PLayer.OutDim(1), PLayer.OutDim(2), PLayer.FNum, numImages);
    % PLayer.poolLocation=ones(1, PLayer.OutDim(1)*PLayer.OutDim(2), PLayer.FNum, numImages);

    switch PLayer.poolMethod
        case {'mean', 'MEAN', 'Mean'}
            % Mean Pooling
            parfor imageNum = 1:numImages
                for featureNum = 1:numFilters
                    featuremap = convolvedFeatures(:, :, featureNum, imageNum);
                    pooledFeaturemap = conv2(featuremap, ones(PLayer.poolDim(1), PLayer.poolDim(2)) / (PLayer.poolDim(1) * PLayer.poolDim(2)), 'valid');
                    pooledFeatures(:, :, featureNum, imageNum) = pooledFeaturemap(1:PLayer.poolDim(1):end, 1:PLayer.poolDim(2):end);
                    % clear pooledFeaturemap;
                end
            end

        case {'max', 'MAX', 'Max'}
            % Max Pooling
            poolLocation = zeros(1, PLayer.OutDim(1) * PLayer.OutDim(2), PLayer.FNum, numImages);

            parfor imageNum = 1:numImages
                for featureNum = 1:numFilters
                    temp = im2col(convolvedFeatures(:, :, featureNum, imageNum), [PLayer.poolDim(1), PLayer.poolDim(2)], 'distinct');
                    [m, i] = max(temp, [], 1);
                    poolLocation(1, :, featureNum, imageNum) = i;
                    pooledFeatures(:, :, featureNum, imageNum) = reshape(m, PLayer.OutDim(1), PLayer.OutDim(2));
                    % clear temp, m;
                end
            end

            PLayer.poolLocation = poolLocation;
            
        otherwise
            error('Unknown Pooling Method!');
    end

end