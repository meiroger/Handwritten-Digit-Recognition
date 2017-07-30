function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This works regardless if z is a matrix or a
%   vector, returning the gradient for each element.

g = zeros(size(z));


% =============================================================
g = sigmoid(z).*(1-sigmoid(z));

% =============================================================

end
