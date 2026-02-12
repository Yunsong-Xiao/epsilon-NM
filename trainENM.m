function [sigmaSquared] = trainENM(train_data)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% trainENM - Training process for the ¦Å-Neighborhood Model.
%
% This function applies an iterative algorithm to compute the parameter sigma^2 
% of the probability density function ¦Ä for a given training dataset. The goal
% is to estimate the model parameters based on the training data.
%
% Inputs:
%   train_data - Normalized training dataset, where:
%                * Each row represents a sample (training instance).
%                * The last column contains labels (target outputs), which are
%                  consecutive integers starting from 1.
%                * The remaining columns contain feature values for each sample.
%
% Outputs:
%   sigmaSquared - The final estimated value of sigma^2 learned through the 
%                  training process, representing the model's parameter for 
%                  the probability density function ¦Ä.
%
% Notes:
%   - The class labels in the dataset must be integers starting from 1 (not 0).
%   - The dataset should be normalized before input.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
    % Data Preprocessing: Extract features and labels from the input data
    normalized_features = train_data(:, 1:end-1); 
    labels = train_data(:, end); 
    classes = unique(labels);
    numClasses =length(classes);
    num_samples = size(normalized_features, 1); 

    % k-NN Search: Find nearest neighbors for each sample
    [neighbors_idx, dist_neighbors] = knnsearch(normalized_features, normalized_features, 'dist', 'euclidean', 'k', num_samples);

    % Build Neighbor Labels Matrix
    neighbor_labels_matrix = labels(neighbors_idx);

    % Calculate average distance (d¦Î) between nearest neighbors
    avg_neighbor_distances = (dist_neighbors(:, 1:end-1) + dist_neighbors(:, 2:end)) / 2;  
    avg_neighbor_distances(:, end+1) = dist_neighbors(:, end); 

    % Compute conditional probability matrix (pxk)
    pxk = zeros(num_samples, num_samples);  
    for i = 1:num_samples
        neighbor_labels = neighbor_labels_matrix(i, 2:end);
        matches = (neighbor_labels == labels(i, 1));  
        cumulative_matches = cumsum(matches);
        pxk(i, 2:end) = cumulative_matches ./ (1:(num_samples - 1));  
    end

    % Assign constant probability for k = 1
    pxk(:, 1) = 1 / numClasses;  % Handle k = 0 case

    % Calculate the first match position (c)
    first_match_positions = inf(num_samples, 1);  
    for i = 1:num_samples
        matches = (neighbor_labels_matrix(i, 2:end) == labels(i, 1));  
        if any(matches)
            first_match_positions(i) = find(matches, 1, 'first'); 
        end
    end
        
    % Calculate frequency-based probability matrix (pkx1)
    pkx1 = zeros(num_samples, num_samples);  
    [unique_positions, ~, position_counts] = unique(first_match_positions);  
        
    % Get the valid indices in unique_positions
    valid_positions = unique_positions(unique_positions > 0);
    % disp('Valid positions before filtering Inf:');
    % disp(valid_positions);

    % Ensure valid_positions does not contain Inf
    valid_positions = valid_positions(isfinite(valid_positions));
    % disp('Filtered valid_positions (after removing Inf):');
    % disp(valid_positions);

    % Ensure that the length of frequencies matches the length of valid_positions
    frequencies = accumarray(position_counts, 1) / num_samples * 100;
    frequencies = frequencies(1:length(valid_positions)); 
    % disp('Length of valid_positions:');
    % disp(length(valid_positions));
    % disp('Length of frequencies:');
    % disp(length(frequencies));

    if length(frequencies) ~= length(valid_positions)
        error('Length mismatch: frequencies and valid_positions must have the same length.');
    end

    % Check if valid_positions exceeds the range of pkx1
    if any(valid_positions > size(pkx1, 2))
        error('Some indices in valid_positions exceed the number of columns in pkx1.');
    end

    % disp('Size of pkx1:');
    % disp(size(pkx1));

    % Insert frequency values into pkx1
    for i = 1:num_samples
        if ~isempty(valid_positions) 
            pkx1(i, valid_positions) = frequencies;
        end
    end

    % Initialize Iteration Variables
    max_iterations = 50;  
    log_distances = zeros(num_samples, num_samples);  
    summed_pkxdx = zeros(max_iterations, 1);  
    sigma2_values = zeros(max_iterations, 1); 
    Q_values = zeros(max_iterations, 1); 
    
    % Start model training with iterative updates
    iteration = 1;

    while iteration <= max_iterations
        % Step 1: Update the parameter of probability density function (sigma^2)
        pkxdx = pkx1 .* avg_neighbor_distances;  
        summed_pkxdx(iteration) = sum(pkxdx(:));  
        sigma2_values(iteration) = summed_pkxdx(iteration) / (2 * num_samples);  % Calculate sigma^2

        % Step 2: Update probability matrix (pkx1)
        exp_vals = exp(-dist_neighbors / (2 * sigma2_values(iteration)));  
        m = diff([exp_vals, zeros(num_samples, 1)], 1, 2);  
        pkx = m .* pxk; 
        normalization_factors = sum(pkx, 2);  
        pkx1 = pkx ./ repmat(normalization_factors, 1, size(pkx, 2));  % Calculate pkx1

        % Compute Q function (objective function)
        log_distances(:, 1:end-1) = log(dist_neighbors(:, 2:end) - dist_neighbors(:, 1:end-1)) - log(2 * sigma2_values(iteration));
        log_distances(:, end) = dist_neighbors(:, end) - log(2 * sigma2_values(iteration));
        logmid = log_distances - avg_neighbor_distances / (2 * sigma2_values(iteration));  
        Q_function = pkx1 .* logmid;  % Element-wise multiplication of pkx1 and logmid
        Q_values(iteration) = sum(sum(Q_function));  
        
        % Increment iteration counter
        iteration = iteration + 1;
        % Output progress every 10 iterations
        if mod(iteration, 10) == 0
            disp(['Iteration ', num2str(iteration), ' / ', num2str(max_iterations)]);
            % disp(['Sigma^2: ', num2str(sigma2_values(iteration))]);
            % disp(['Q value: ', num2str(Q_values(iteration))]);
        end
    end

    % Return the final sigma^2 values after training
    % sigmaSquared = sigma2_values;  % Return all sigma2 values computed
    sigmaSquared = sigma2_values(end,1);
end
