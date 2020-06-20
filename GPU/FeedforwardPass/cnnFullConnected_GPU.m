function fullyConnected=cnnFullConnected_GPU(FLayer, images)

fullyConnected=bsxfun(@plus, FLayer.W*images, FLayer.B);