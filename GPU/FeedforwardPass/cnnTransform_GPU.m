function out = cnnTransform_GPU(images, CLayer)

    switch CLayer.TName
        case {'FFT', 'fft'}
            out = fft(fft(images, [], 1), [], 2);
        case {'DWT', 'dwt'}
            % Not installed
        case {'PCA', 'pca'}
            numImage = size(images, 4);
            numFilter = size(images, 3);
            % out=gpuArray.zeros(size(images), 'single');
            out_ = zeros(size(images), 'single');
            images_ = gather(images);

            parfor inum = 1:numImage
                for iflt = 1:numFilter
                    [U, S, V] = svd(images_(:, :, iflt, inum));
                    U = U(:, CLayer.PCADim);
                    S = S(CLayer.PCADim, :);
                    PCAImage = U * S * V';
                    out_(:, :, iflt, inum) = single(PCAImage);
                end
            end

            out = gpuArray(out_);
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
            [~, out] = cnnPool_GPU(CLayer, images);
        case {'MEANPOOL', 'meanpool'}
            CLayer.poolMethod = 'mean';
            [~, out] = cnnPool_GPU(CLayer, images);
        case {'LOWPASS', 'lowpass'}
        case {'HIGHPASS', 'highpass'}
        otherwise
            error('Unknown Transformation Type!');
    end

end