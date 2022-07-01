function [train, validate] = ...
    Billen_Moritz_splitTrainValidate(data, idx, train_ratio)
% BILLEN_MORITZ_SPLITTRAINVALIDATE splits data set into train and test sets
%       Randomize data and cut data as well as labels into train and test
%       sets.
% Input:
%       - data: (n_samples,n_features) array with data set
%       - idx: (n_samples,1) array with labels for each sample
%       - train_ratio: Float value between 0 and 1 which percentage of data
%       should be attributed to the train set.
% Output:
%       - train: struct with fields data and idx containing the samples
%       assigned to the train set
%       - validate: struct with fields data and idx containing the samples
%       assigned to the validate set

%% Input Validation

arguments
    data (:,:) {mustBeNumeric}
    idx (:,1) {mustBeInteger}
    train_ratio (1,1) {mustBeNumeric}
end

% Check if dimensions line up
assert(size(data, 1) == size(idx, 1), ...
    "data and idx must have the same amount of samples")

% Check if train_ratio is valid
assert(0 < train_ratio & train_ratio <= 1, "Train ratio must be in [0,1]")

% Get n_samples from data set
[n_samples, ~] = size(data);

% Get new random order of indices
new_idx = randperm(n_samples);

% Get split index
split_idx = floor(n_samples * train_ratio);

% Write data fields
train.data = data(new_idx(1:split_idx), :);
validate.data = data(new_idx(split_idx + 1:end), :);

% Write label array fields
train.idx=idx(new_idx(1:split_idx));
validate.idx=idx(new_idx(split_idx+1:end));