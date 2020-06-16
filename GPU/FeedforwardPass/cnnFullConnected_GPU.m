function fullyConnected=cnnFullConnected_GPU(FLayer, images)

fullyConnected=bsxfun(@plus, gather(gpuArray(FLayer.W)*gpuArray(images)), FLayer.B);