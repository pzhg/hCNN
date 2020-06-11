function out=cnnTransform(images, CLayer)

switch CLayer.TName
    case 'FFT'
        out=fft(fft(images, [], 1), [], 2);
    case 'DWT'
        % Not installed
    case 'PCA'
        numImage=size(images, 4);
        numFilter=size(images, 3);
        if CLayer.useGPU==1
            out=single(gpuArray.zeros(size(images)));
        else
            out=single(zeros(size(images)));
        end
        parfor inum=1:numImage
            for iflt=1:numFilter
                [U, S, V]=svd(double(images(:, :, iflt, inum)));
                U=U(:, CLayer.PCADim);
                S=S(CLayer.PCADim, :);
                PCAImage=U*S*V';
                out(:, :, iflt, inum)=single(PCAImage);
            end
        end
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
        [~, out]=cnnPool(CLayer, images);
    case 'MEANPOOL'
        CLayer.poolMethod='mean';
        [~, out]=cnnPool(CLayer, images);
end
