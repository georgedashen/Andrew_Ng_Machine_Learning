%% Multiclass Logistic Regression
%  Use an artificial dataset that classified into 3 sets with two
%  attributes
%  A method described as one-vs-all (one-vs-rest)
%% Initialization
clear ; close all; clc

%% Create data
x1 = rand(50,1)*10;
y1 = rand(50,1)*10+10;
x2 = rand(50,1)*10+5;
y2 = rand(50,1)*10;
x3 = rand(50,1)*10+15;
y3 = rand(50,1)*10+5;
x = [x1;x2;x3];
y = [y1;y2;y3];
X = [x,y];
y = [ones(50,1);ones(50,1)*2;ones(50,1)*3];

figure;
plot(x1,y1,'b+','MarkerSize',7);
hold on
plot(x2,y2,'k*','MarkerSize',7);
plot(x3,y3,'ro','MarkerSize',7);

%% Iteration
[m, n] = size(X);

% Add intercept term to x and X_test
X = [ones(m, 1) X];

% Initialize fitting parameters
initial_theta = zeros(n + 1, 3);

options = optimset('GradObj', 'on', 'MaxIter', 400);

for i=1:3
    Y = y;
    Y(find(y~=i)) = 0;
    Y(find(y==i)) = 1;
    [theta(:,i), cost] = ...
	fminunc(@(t)(costFunction(t, X, Y)), initial_theta(:,i), options);
end

p = predictMulti(theta,X);
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);

for i=1:3
    plot_x = [min(X(:,2))-2,  max(X(:,2))+2];
    plot_y = (-1./theta(3,i)).*(theta(2,i).*plot_x + theta(1,i));
    plot(plot_x, plot_y);
end
legend('class1', 'class2', 'class3')
axis([0 25 0 20])
hold off

%% Regularization for Multiclass
X = mapFeature(X(:,2), X(:,3));
rng(2019);
initial_theta = randn(size(X, 2), 3)*1e-8;
theta = zeros(size(initial_theta));
lambda = 10;
options = optimset('GradObj', 'on', 'TolX',1e-10, 'MaxIter', 1000);

for i=1:3
    Y = y;
    Y(find(y~=i)) = 0;
    Y(find(y==i)) = 1;
    [theta(:,i), J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t,X,Y,lambda)), initial_theta(:,i), options);
end

p = predictMulti(theta,X);
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);

figure;
plot(x1,y1,'b+','MarkerSize',7);
hold on
plot(x2,y2,'k*','MarkerSize',7);
plot(x3,y3,'ro','MarkerSize',7);

for c = 1:size(theta,2)
    u = linspace(0, 25, 50);
    v = linspace(0, 20, 50);
    z = zeros(length(u), length(v));

    for i = 1:length(u)
        for j = 1:length(v)
            z(i,j) = mapFeature(u(i), v(j))*theta(:,c);
        end
    end
    z = z'; 
    contour(u, v, z, [0, 0], 'LineWidth', 2)
end

hold off