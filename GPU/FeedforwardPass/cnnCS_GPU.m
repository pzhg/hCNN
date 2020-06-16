function fltOutput=cnnCS_GPU(images, CLayer)

numImages=size(images, 4);
numFilter=size(images, 3);
fltOutput=zeros(CLayer.OutDim(1), CLayer.OutDim(2), numFilter, numImages, 'single');

parfor inum=1:numImages
    for iflt=1:numFilter
        image=images(:, :, iflt, inum);
        CSImage=gpuArray(CLayer.A)*gpuArray(image(:));
        fltOutput(:, 1, iflt, inum)=gather(CSImage);
    end
end