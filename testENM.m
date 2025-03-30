function accuracy = testENM(trainData, testData, numClasses, sigmaSquared)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% testENM - Testing process for the epsilon-Neighborhood Model. 
%
% This function implements the epsilon-Neighborhood Model for classifying data, leveraging the k-Nearest Neighbors (k-NN) algorithm and Semantic Cell Models.
% It evaluates the model's performance on a test dataset using the trained model and the provided parameters.
% 
% Inputs:
%   trainData    - The normalized training dataset, where the last column contains the class labels (target outputs).
%   testData     - The normalized test dataset, where the last column contains the class labels (target outputs).
%   numClasses   - The number of distinct categories (classes) in the dataset.
%   sigmaSquared - The parameter sigma^2 for the probability density function delta learned during the training process.
%
% Outputs:
%   accuracy     - The classification accuracy, defined as the proportion of correctly predicted labels in the test dataset.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   % Extract features and labels
    trainFeatures = trainData(:, 1:end-1);  
    testFeatures = testData(:, 1:end-1);    
    numTrainData = size(trainFeatures, 1);  
    numTestData = size(testFeatures, 1);    
    trainLabels = trainData(:, end);        
    testLabels = testData(:, end);          

    % Perform k-NN search to find nearest neighbors
    [indexNN, distNN] = knnsearch(trainFeatures, testFeatures, 'dist', 'euclidean', 'k', numTrainData);

    % Compute the neighborhood degree matrix 'u'
    % 'u' contains the exponential decay of distances, with the first column set to 1
    u = ones(numTestData, numTrainData + 1);  
    u(:, 2:end) = exp(-distNN / (2 * sigmaSquared)); 

    % Calculate the 'mx' matrix (weight matrix based on neighborhood differences)
    % mx(:, j) = u(:, j) - u(:, j+1), except for the last column
    mx = zeros(size(u)); 
    mx(:, 1:end-1) = u(:, 1:end-1) - u(:, 2:end);  
    mx(:, end) = u(:, end);  

    % Get the class labels of the k-NN for each test sample
    indexNNClass = trainLabels(indexNN); 

    % Calculate the probability matrix 'pck'
    % pck(:, k+1, c) contains the probability of each test sample belonging to class 'c'at the k-th nearest neighbor
    pck = zeros(numTestData, numTrainData + 1, numClasses);  
    
    for i = 1:numTestData
        for c = 1:numClasses
            class_mask = (indexNNClass(i, :) == c);
            class_cumsum = cumsum(class_mask);
            pck(i, 2:numTrainData, c) = class_cumsum(1:numTrainData-1) ./ (1:numTrainData-1);
        end
    end

    % Assign prior probability for each class (uniform prior)
    pck(:, 1, :) = 1 / numClasses;  

    % Compute the final probability matrix 
    weightMatrix3D = zeros(size(mx, 1), size(mx, 2), numClasses);

    for c = 1:numClasses
        weightMatrix3D(:, :, c) = mx; 
    end

    % Compute weighted probabilities for each class
    weightedProbabilities = weightMatrix3D .* pck;  

    % Get the final score for each class
    finalScores = sum(weightedProbabilities, 2);

    % Determine the predicted class for each test sample
    [~, predictedLabels] = max(finalScores, [], 3);  

    % Calculate classification accuracy
    correctPredictions = sum(predictedLabels == testLabels); 
    accuracy = correctPredictions / numTestData;  
end