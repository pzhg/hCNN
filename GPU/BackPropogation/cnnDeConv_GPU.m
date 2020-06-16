function DeltaPooled=cnnDeConv_GPU(CLayer, DeltaConv)

numFilters1=size(CLayer.W, 3);
numFilters2=size(CLayer.W, 4);
numImages=size(DeltaConv, 4);
ConvDim=CLayer.OutDim+CLayer.FDim-1;

DeltaPooled=gpuArray.zeros(ConvDim(1), ConvDim(2), numFilters1, numImages, 'single');      
parfor i=1:numImages
    for f1=1:numFilters1
        for f2=1:numFilters2
            DeltaPooled(:, :, f1, i)=DeltaPooled(:, :, f1, i)+convn(DeltaConv(:, :, f2, i), CLayer.W(:, :, f1, f2), 'full');
        end
    end
end