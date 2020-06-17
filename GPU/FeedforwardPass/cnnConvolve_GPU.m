function convolvedFeatures=cnnConvolve_GPU(CLayer, images)

numFilters1=size(CLayer.W, 3);
numImages=size(images, 4);
numFilters=CLayer.FNum;

convolvedFeatures=zeros(CLayer.OutDim(1), CLayer.OutDim(2), CLayer.FNum, numImages, 'single');
% g_convolvedFeatures=gpuArray.zeros(CLayer.OutDim(1), CLayer.OutDim(2), CLayer.FNum, numImages, 'single');
% g_images=gpuArray(images);
% g_W=gpuArray(CLayer.W);
% g_B=gpuArray(CLayer.B);

parfor i=1:numImages
    for fil2=1:numFilters
        convolvedImage=zeros(CLayer.OutDim, 'single');
        for fil1=1:numFilters1
            filter=gpuArray(rot90(CLayer.W(:, :, fil1, fil2), 2));
            im=gpuArray(images(:, :, fil1, i));
            convolvedImage=convolvedImage+gather(conv2(im, filter, 'valid'));
        end
        convolvedImage=bsxfun(@plus, convolvedImage, CLayer.B(fil2));
        convolvedFeatures(:, :, fil2, i)=convolvedImage;
%         clear filter, convolvedImage;
    end
end

% convolvedFeatures=gather(g_convolvedFeatures);