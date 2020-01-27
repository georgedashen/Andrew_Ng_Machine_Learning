%% Compute prediction using lWR (locally weighted regression)
%  theta_j+1 = theta_j - alpha/m * sum (h_theta(xi)-yi)*x_j (i from 1 to m)
%  h_theta(x) = X*theta
%  w_i = exp(-(x_i-x)^2/(2*tao^2)) where tao = 0.5
%% Initialization
clear ; close all; clc
fprintf('Loading data ...\n');

%% Load Data
data = load('ex1data1.txt');
X = data(:, 1);
y = data(:, 2);
m = length(y);
plotData(X, y);

%% LWR
X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters

% Some gradient descent settings
iterations = 1500;
alpha = 0.01;
tao = 0.5;
% x = [1, 3.5];
x = [1, 7];

w = exp(-(X(:,2)-x(2)).^2./(2*tao^2));

for iter = 1:iterations
    j1 = theta(1) - alpha/m*(w.*(X*theta-y))'*X(:,1);
    j2 = theta(2) - alpha/m*(w.*(X*theta-y))'*X(:,2);
    theta = [j1;j2];
end

% Plot the linear fit
hold on; % keep previous plot visible
plot(X(:,2), X*theta, '-')
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure

predict = x * theta