function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Part 1: Cost

% Step 1: Solving a3 = hypothesis for this nn

% Add ones to x 
a1 = [ones(m, 1) X]; % Mx401

% Solving a2
a2 = sigmoid(a1*Theta1'); % Mx401 401x25 = Mx25

% Add ones to a2
a2 = [ones(m, 1) a2]; % Mx26

% Solving a3
a3 = sigmoid(a2*Theta2'); % Mx26 26x10 = Mx10

% Step 2: Solve cost using for-loop, K = num_labels

% Setup a variable
K = 0;
cost = 0;

for K = 1:num_labels
    cost = cost+((y==K)'*log(a3(:,K))...
        +(1-(y==K))'*log(1-a3(:,K))); % 1xM Mx1 = 1x1
end

J = cost*(-1)/m;

% Calculate regularization term

% Not regularizing bias, thus replace first column as 0
Temp1 = Theta1;
Temp1(1:end,1) = 0;

Temp2 = Theta2;
Temp2(1:end,1) = 0;

% Calculate Reg = 'regularization term' using formula
Reg = (lambda/(2*m))*((sum(sum(Temp1.^2)))+(sum(sum(Temp2.^2))));

% Add Reg to J
J = J + Reg;
    

% Part 2: Implement Backpropagation algorithm 

% for-loop for Step 1~4
for t = 1:m
    
    % Step 1
    
    % Add one to X
    a_1 = X(t,:); % 1x400
    a_1 = a_1'; % 400x1
    a_1 = [1; a_1]; % 401x1
    
    % Solve a_2
    z_2 = Theta1*a_1; % 25x401 401x1 = 25*1
    a_2 = sigmoid(z_2);
    
    % Add one to a_2
    a_2 = [1; a_2]; % 26x1
    
    % Solve a_3
    z_3 = Theta2*a_2; % 10x26 26x1 = 10x1
    a_3 = sigmoid(z_3);
    
    % Step 2
    
    % Setup variables
    K = 0;
    delta_3 = zeros(size(a_3));
    
    for K = 1:num_labels
        delta_3(K) = a_3(K)-(y(t)==K);
    end
    
    % Step 3
    
    z_2 = [0; z_2]; % 26x1
    delta_2 = Theta2'*delta_3.*sigmoidGradient(z_2); % 26x10 10x1
    delta_2 = delta_2(2:end); % 25x1 remove 1st column (for bias term)
    
    % Step 4
    
    Theta1_grad = Theta1_grad+delta_2*a_1'; % 25x1 1x401 = 25x401
    Theta2_grad = Theta2_grad+delta_3*a_2'; % 10x1 1x26 = 10x26
    
end

% Step 5

% Divide gradients by m
Theta1_grad = Theta1_grad/m;
Theta2_grad = Theta2_grad/m;

% Add regularization terms
Temp1 = Theta1;
Temp2 = Theta2;
Temp1(1:end, 1) = 0;
Temp2(1:end, 1) = 0;

Theta1_grad = Theta1_grad+(lambda/m).*Temp1;
Theta2_grad = Theta2_grad+(lambda/m).*Temp2;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
