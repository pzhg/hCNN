function probs = cnnSoftMax(input)

    F = -max(input);
    probs = exp(input + F);
    sumProbs = sum(probs, 1);
    probs = bsxfun(@times, probs, 1 ./ sumProbs);

end