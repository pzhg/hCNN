function delta_out = cnnDeFullConnected(FLayer, delta_in)

    delta_out = FLayer.W' * delta_in;

end