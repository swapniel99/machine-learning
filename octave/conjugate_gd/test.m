clear ; close all; clc

data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);

[_ mu sigma] = featureNormalize(X);

m=length(y);

%temp = loader();	% Load the training data once in form [y,X]
%y = temp(:,1);
%X = temp(:,2:end);
%m = length(y); 

X = [ones(m, 1) X];

initialTheta = zeros(3, 1);
options = optimset('GradObj','on','MaxIter','100');
[optTheta, functionVal, exitFlag] = fminunc(@costFunction,initialTheta,options);

fprintf('Theta computed from conjugate gradient: \n');
fprintf(' %f \n', optTheta);
fprintf('\n');

price = [1,([1650 3]-mu)./sigma]*optTheta;

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using conjugate gradient):\n $%f\n'], price);

fprintf('Solving with normal equations...\n');


theta = normalEqn(X, y);

fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');

price = [1 1650 3]*theta;

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n $%f\n'], price);

