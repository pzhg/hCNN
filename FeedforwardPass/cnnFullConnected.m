function fullyConnected=cnnFullConnected(FLayer, images)

fullyConnected=bsxfun(@plus, FLayer.W*images, FLayer.B);