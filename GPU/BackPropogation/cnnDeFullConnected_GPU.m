function delta_out=cnnDeFullConnected_GPU(FLayer, delta_in)

delta_out=gather(gpuArray(FLayer.W')*gpuArray(delta_in));