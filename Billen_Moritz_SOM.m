function [out,weights] = Billen_Moritz_SOM(data,options)
% BILLEN_MORITZ_SOM trains a self organiing map
%   Kohonen SOM is fitted to the dataset. Grid is initialized to be n_grid
%   on each side length. Trained for n_epochs epochs on every single sample
%   in data. Neighborhood parameter sigm decays from sigm_init with the
%   parameter sigm_decay. Same for the learning rate lr (lr_init, lr_decay)

    %% Check inputs
    arguments
        data (:,2) {mustBeNumeric}
        options.n_grid (1,1) {mustBeInteger} = 5;
        options.n_epochs (1,1) {mustBeInteger} = 10;
        options.sigm_init (1,1) {mustBeNumeric} = 2;
        options.sigm_decay (1,1) {mustBeNumeric} = 5;
        options.lr_init (1,1) {mustBeNumeric} = 0.5;
        options.lr_decay (1,1) {mustBeNumeric} = 5;
    end

    [~, n_features] = size(data);

    % Paste input data into shorter variables
    n_grid = options.n_grid;
    n_epochs = options.n_epochs;
    sigm_init = options.sigm_init;
    sigm_decay = options.sigm_decay;
    lr_init = options.lr_init;
    lr_decay = options.lr_decay;

    % Randomly initialize weights
    weights = randn(n_grid,n_grid,n_features);
    
    % Define function handle for manhattan distance in 2D
    n_dist = @(i_1,j_1,i_2,j_2) abs(i_1 - i_2) + abs(j_1 - j_2);

    % Indice matrices for Manhattan calculation
    [c_x,c_y] = meshgrid(1:n_grid,1:n_grid);
    
    % Initialize sigm and lr
    sigm = sigm_init;
    lr=lr_init;    

    % Iterate over all epochs
    for epoch = 1:n_epochs

        % Iterate over sample vectors
        for d = data'

            % Get euclidean distance to current sample point d
            res = sqrt(sum((reshape(d, 1, 1, 2)-weights).^2, 3));

            % Get index of minimal distance
            [~,i_m]=min(res,[],"all");
            
            % Split index into x and y
            i_y=fix((i_m-1)/n_grid)+1;
            i_x=mod((i_m-1),n_grid)+1;
            
            % Manhattan distance from winner node to all other nodes
            dist=n_dist(i_x,i_y,c_x,c_y);

            % Compute weights which weights are affected how much
            h = exp(-dist.^2/2/sigm^2);

            % Update weights
            weights(:,:,1) = weights(:,:,1) + lr * h .* (d(1)-weights(:,:,1));
            weights(:,:,2) = weights(:,:,2) + lr * h .* (d(2)-weights(:,:,2));
        end

        % Update learning rate and sigm
        lr = lr_init * (1-exp(-epoch/lr_decay));
        sigm = sigm_init * exp(-epoch/sigm_decay);

        % Print progress
        fprintf("Iteration %d/%d\n",epoch,n_epochs)
    end
    
    % Output function handle
    out = @(x) 1./(1+exp(-sum(pagemtimes(reshape(x,1,1,2),weights),3)));
end

