function convolvedFeatures=cnnConvolve(CLayer, images)

numFilters1=size(CLayer.W, 3);
numImages=size(images, 4);
numFilters=CLayer.FNum;
if CLayer.useGPU==1
    convolvedFeatures=single(gpuArray.zeros(CLayer.OutDim(1), CLayer.OutDim(2), CLayer.FNum, numImages));
else
    convolvedFeatures=single(zeros(CLayer.OutDim(1), CLayer.OutDim(2), CLayer.FNum, numImages));
end

parfor i=1:numImages
    for fil2=1:numFilters
        if CLayer.useGPU==1
            convolvedImage=single(gpuArray.zeros(CLayer.OutDim));
            for fil1=1:numFilters1
                filter=gpuArray(rot90(squeeze(CLayer.W(:, :, fil1, fil2)), 2));
                im=squeeze(images(:, :, fil1, i));
                convolvedImage=convolvedImage+single(conv2(im, filter, 'valid'));
            end
        else
            convolvedImage=single(zeros(CLayer.OutDim));
            for fil1=1:numFilters1
                filter=rot90(squeeze(CLayer.W(:, :, fil1, fil2)), 2);
                im=squeeze(images(:, :, fil1, i));
                convolvedImage=convolvedImage+single(conv2(im, filter, 'valid'));
            end
        end
        convolvedImage=bsxfun(@plus, convolvedImage, CLayer.B(fil2));
        convolvedFeatures(:, :, fil2, i)=convolvedImage;
%         clear filter, convolvedImage;
    end
end