function probs=cnnSoftMax(input)

probs=exp(input);
sumProbs=sum(probs, 1);
probs=bsxfun(@times, probs, 1./sumProbs);