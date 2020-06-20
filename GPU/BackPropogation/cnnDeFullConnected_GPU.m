function delta_out=cnnDeFullConnected_GPU(FLayer, delta_in)

delta_out=FLayer.W'*delta_in;