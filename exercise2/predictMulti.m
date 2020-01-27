function p = predictMulti(theta, X)
%PREDICT Predict whether the class using multiple logistic 
%regression by one-vs-rest method

m = size(X, 1); % Number of training examples
p = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters. 
%               You should set p to a vector of 0's and 1's
%
P = X*theta;
for i = 1:m
    p(i) = find(P(i,:) == max(P(i,:)));
end

% =========================================================================


end
