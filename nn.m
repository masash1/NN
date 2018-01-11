%% Robot Intelligence Final Assignment - Neural Network Learning
%  1526708 Masashi Kaneko
%% Functions used in this script
% # displayData.m
% # fmincg.m
% # nnCostFunction.m
% # predict.m
% # randInitializeWeights.m
% # sigmoid.m
% # sigmoidGradient.m

%% Initialization
clear; close all; clc

%% Setup the parameters
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that "0" has been mapped to label 10)

%% Loading and Visualizing Data

% Load Data
fprintf('Loading and Visualizing Data ...\n')

load('data.mat');
m = size(X, 1);

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

% Separate the original dataset for training and testing
X_train = X(1:4900, :);
y_train = y(1:4900, :);

X_test = X(4901:end, :);
y_test = y(4901:end, :);

%% Initializing Pameters

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%% Training NN

fprintf('\nTraining Neural Network... \n')

options = optimset('MaxIter', 500);

lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X_train, y_train, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

%% Visualize Weights
%  Visualize what the neural network is learning by displaying the 
%  hidden units to see what features they are capturing in the data.

fprintf('\nVisualizing Neural Network... \n')

displayData(Theta1(:, 2:end)); % Theta1 = 25x401(nfeatures+1)

%% Implement Predict

pred = predict(Theta1, Theta2, X_test);

fprintf('\nTesting Set Accuracy: %f\n', mean(double(pred == y_test)) * 100);
