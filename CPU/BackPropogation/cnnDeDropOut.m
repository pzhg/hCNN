function delta_out = cnnDeDropOut(DLayer, delta_in)

    delta_out = delta_in .* DLayer.location;
    delta_out = delta_out / (1 - DLayer.rate);

end