function [data, idx] = Billen_Moritz_genGaussianDataset(n_points, ...
    means, options)
% BILLEN_MORITZ_GENGAUSSIANDATASET generates gaussian data clusters
%       Gaussian data is generated based on number of points, mean, and
%       covariance matrix. For an arbitrary amount of clusters to be
%       generated these inputs can be given in the input arrays.
%       Warning: the generated output is not randomized, but grouped by
%       clusters.
% Input:
%       - n_points: (n_clusters,1) array with number of samples to be
%       generated in each cluster
%       - means: (n_clusters,n_features) array with means for each gaussian
%       cluster
%       - options.cov (n_clusters,n_features,n_features) struct with
%       covariance matrices for each cluster
% Output:
%       - data: (n_samples,n_features) array of the samples
%       - idx: (n_samples,1) array with the label indices for each sample
%       point

%% Input Validation

arguments
    n_points    (:,1) {mustBeInteger}
    means        (:,:) {mustBeNumeric}
    
    % Passing struct as positional argument is not possible in matlab
    options.cov (:,1) struct 
end

% Get parameters from input arrays
n_dim = size(means, 2);
n_cluster = size(n_points, 1);

% Check if there is n_cluster numbers of points
assert(size(n_points, 1) == n_cluster)

% Check if there is n_cluster means
assert(size(means, 1) == n_cluster)

% Check if there is n_dim dimensions for mean
assert(size(means, 2) == n_dim)

% Check if there is n_cluster covariance matrices
assert(size(options.cov, 1) == n_cluster)

% Check if the dimension of each covariance matrix is n_dim x n_dim
for i = 1:n_cluster
    assert(all(size(options.cov(i).mat) == n_dim))
end

%% Fill output arrays

% Initialize output arrays
data = zeros(sum(n_points), n_dim);
idx = zeros(sum(n_points), 1);

% Initialize write indices
start_idx = 1;
end_idx = 0;

% Loop over number of clusters
for i=1:n_cluster

    % Increase end_idx before writing data
    end_idx = end_idx + n_points(i);

    % Write random data
    curr_idx = start_idx:end_idx;

    % Use MVNRND to generate data based on means and covariance matrices
    data(curr_idx, :) = ...
        mvnrnd(means(i, :), options.cov(i).mat, n_points(i));

    % Write index in idx array 
    idx(curr_idx) = i;
    
    % Increase start index after writing data
    start_idx = end_idx + 1;
end