function [dropOut, DLayer] = cnnDropOut(images, DLayer)

    if DLayer.test == 0
        dim = size(images, 1);
        location = binornd(ones(dim, 1), DLayer.rate);
        dropOut = images .* location;
        dropOut = dropOut / DLayer.rate;
        DLayer.location = location;
    else
        dropOut=images;
    end

end
