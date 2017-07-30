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
%   nn_params and converted back into the weight matrices. 
% 
%   The returned parameter grad is a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for the 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Initialize variables
m = size(X, 1);
y_matrix = zeros(size(y,1),num_labels);
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% =========================================================================
id = eye(num_labels);
for i = 1:size(y,1)
  y_matrix(i,:) = id(y(i,1),:); %5000x10
end
 
% Forward Propagation 
a1 = [ones(m,1) X]; %mx401
z2 = a1*Theta1'; %mx401 * 401x25 = mx25
a2 = [ones(m,1) sigmoid(z2)]; %mx26
z3 = a2*Theta2'; %mx26 * 26x10 = mx10
a3 = sigmoid(z3); %mx10

% Trace is sum of diagonal (double sum)
J = (1/m)*trace((-y_matrix'*log(a3)-(1-y_matrix)'*log(1-a3))); %10x5000 * mx10 = 10x10

%Adding regularization term to cost function
Theta1(:,1) = 0;
Theta2(:,1) = 0;
r1 = sum(sum(Theta1.^2));
r2 = sum(sum(Theta2.^2));
regular = (lambda/(2*m))*(r1+r2);
J = J + regular;

% Backpropagation
d3 = a3 - y_matrix;
d2 = (d3*Theta2(:,2:end)).*sigmoidGradient(z2);
Delta1 =  d2'*a1;
Delta2 = d3'*a2;
% Scaling and adding regularization
Theta1_grad = Delta1./m + (lambda/m).*Theta1; 
Theta2_grad = Delta2./m + (lambda/m).*Theta2;
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
