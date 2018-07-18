function acc=cnnTestData(cnn, VX, VY, numImages)
% Validate CNN Accuracy
%   VData: validation data, [x-dim, y-dim, channel-num, data-count]
%   VLabel: validation label, [1, data-count]
%   numImages: number of images that want to validate

images=VX(:, :, :, 1:numImages);
clear VData;
mb_labels=VY(:, 1:numImages);
clear VLabel;

[~, OutData]=cnnFeedForward(cnn, images);
[~, preds]=max(OutData{cnn.LNum}, [], 1);
acc=sum(preds==mb_labels)/numImages;