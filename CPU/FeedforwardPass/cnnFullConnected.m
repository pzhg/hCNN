function fullyConnected=cnnFullConnected(FLayer, images)

fullyConnected=bsxfun(@plus, FLayer.W*squeeze(images), FLayer.B);