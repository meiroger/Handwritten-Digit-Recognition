function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
% Initialize p
p = zeros(size(X, 1), 1); 

% =========================================================================
a1 = [ones(m, 1) X]; % mx401
z2 = a1*Theta1'; % mx401 * 401x25 = mx25
a2 = [ones(size(z2,1),1) sigmoid(z2)]; % mx26;
z3 = a2*Theta2'; % mx10;
a3 = sigmoid(z3);
[v p] = max(a3,[],2);
% =========================================================================
end
