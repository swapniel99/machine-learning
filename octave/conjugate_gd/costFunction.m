function [jVal,gradient] = costFunction(theta)

persistent temp = loader();	% Load the training data once in form [y,X]
persistent y = temp(:,1);
persistent m = length(y); 
persistent X = [ones(m,1),temp(:,2:end)];
persistent n = length(theta);

jVal = sum((X*theta-y).^2)/(2*m);  % Cost function
gradient = sum(repmat(X*theta-y,1,n).*X)'./m; % Partial derivatives

end
