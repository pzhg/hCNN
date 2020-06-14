function convolvedFeatures=cnnConvolve_GPU(CLayer, images)

numFilters1=size(CLayer.W, 3);
numImages=size(images, 4);
numFilters=CLayer.FNum;
convolvedFeatures=single(gpuArray.zeros(CLayer.OutDim(1), CLayer.OutDim(2), CLayer.FNum, numImages));

parfor i=1:numImages
    for fil2=1:numFilters
        convolvedImage=single(gpuArray.zeros(CLayer.OutDim));
        for fil1=1:numFilters1
            filter=gpuArray(rot90(squeeze(CLayer.W(:, :, fil1, fil2)), 2));
            im=squeeze(images(:, :, fil1, i));
            convolvedImage=convolvedImage+single(conv2(im, filter, 'valid'));
        end
        convolvedImage=bsxfun(@plus, convolvedImage, CLayer.B(fil2));
        convolvedFeatures(:, :, fil2, i)=single(convolvedImage);
%         clear filter, convolvedImage;
    end
end