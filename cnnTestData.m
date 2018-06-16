function acc=cnnTestData(cnn, VData, VLabel, numImages)
% Validate CNN Accuracy
%   VData: validation data, [x-dim, y-dim, channel-num, data-count]
%   VLabel: validation label, [1, data-count]
%   numImages: number of images that want to validate

images=VData(:, :, :, 1:numImages);
clear VData;
mb_labels=VLabel(:, 1:numImages);
clear VLabel;

[~, OutData]=cnnFeedForward(cnn, images);
[~, preds]=max(OutData{cnn.LNum}, [], 1);
acc=sum(preds==mb_labels)/numImages;