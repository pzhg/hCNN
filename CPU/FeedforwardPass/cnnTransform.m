function out = cnnTransform(images, CLayer)

    switch CLayer.TName
        case {'FFT', 'fft'}
            out = fft(fft(images, [], 1), [], 2);
        case {'DWT', 'dwt'}
            % Not installed
        case {'PCA', 'pca'}
            numImage = size(images, 4);
            numFilter = size(images, 3);
            out = zeros(size(images));

            parfor inum = 1:numImage
                for iflt = 1:numFilter
                    [U, S, V] = svd(double(images(:, :, iflt, inum)));
                    U = U(:, CLayer.PCADim);
                    S = S(CLayer.PCADim, :);
                    PCAImage = U * S * V';
                    out(:, :, iflt, inum) = PCAImage;
                end
            end

        case {'ABS', 'abs'}
            out = abs(images);
        case {'ARG', 'arg'}
            out = angle(images);
        case {'REAL', 'real'}
            out = real(images);
        case {'IMAG', 'imag'}
            out = imag(images);
        case {'MAXPOOL', 'maxpool'}
            CLayer.poolMethod = 'max';
            [~, out] = cnnPool(CLayer, images);
        case {'MEANPOOL', 'meanpool'}
            CLayer.poolMethod = 'mean';
            [~, out] = cnnPool(CLayer, images);
        case {'LOWPASS', 'lowpass'}
        case {'HIGHPASS', 'highpass'}
        otherwise
            error('Unknown Transformation Type!');
    end

end