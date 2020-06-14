function [Wc_grad, bc_grad]=cnnConvGrad_GPU(activationsPooled, DeltaConv)

numImages=size(activationsPooled, 4);
numFilters2=size(DeltaConv, 3);
numFilters1=size(activationsPooled, 3);
ConvDim=size(activationsPooled)-size(DeltaConv)+1;

Wc_grad=single(gpuArray.zeros(ConvDim(1), ConvDim(2), numFilters1, numFilters2));
bc_grad=single(gpuArray.zeros(numFilters2, 1));
parfor fil2=1:numFilters2
    for fil1=1:numFilters1
        for im=1:numImages
            Wc_grad(:, :, fil1, fil2)=Wc_grad(:, :, fil1, fil2)+conv2(activationsPooled(:, :, fil1, im), rot90(DeltaConv(:, :, fil2, im), 2), 'valid');
        end
    end
    temp=DeltaConv(:, :, fil2, :);
    bc_grad(fil2)=single(sum(temp(:)));
%     clear temp;
end