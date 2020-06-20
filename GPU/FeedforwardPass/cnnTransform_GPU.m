function out=cnnTransform_GPU(images, CLayer)

switch CLayer.TName
    case 'FFT'
        out=fft(fft(images, [], 1), [], 2);
    case 'DWT'
        % Not installed
    case 'PCA'
        numImage=size(images, 4);
        numFilter=size(images, 3);
        % out=gpuArray.zeros(size(images), 'single');
        out_=zeros(size(images), 'single');
        images_=gather(images);
        parfor inum=1:numImage
            for iflt=1:numFilter
                [U, S, V]=svd(images_(:, :, iflt, inum));
                U=U(:, CLayer.PCADim);
                S=S(CLayer.PCADim, :);
                PCAImage=U*S*V';
                out_(:, :, iflt, inum)=single(PCAImage);
            end
        end
        out=gpuArray(out_);
    case 'ABS'
        out=abs(images);
    case 'ARG'
        out=angle(images);
    case 'REAL'
        out=real(images);
    case 'IMAG'
        out=imag(images);
    case 'MAXPOOL'
        CLayer.poolMethod='max';
        [~, out]=cnnPool_GPU(CLayer, images);
    case 'MEANPOOL'
        CLayer.poolMethod='mean';
        [~, out]=cnnPool_GPU(CLayer, images);
    case 'LOWPASS'
    case 'HIGHPASS'
end
