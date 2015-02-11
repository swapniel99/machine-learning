function [jVal,gradient] = costFunction(theta)

persistent temp = loader();	% Load the training data once in form [y,X]
persistent y = temp(:,1);
persistent m = length(y); 
persistent X = [ones(m,1),temp(:,2:end)];

jVal = sum((X*theta-y).^2)/(2*m);  % Cost function
gradient = (X'*(X*theta - y))/m; % Partial derivatives

end
