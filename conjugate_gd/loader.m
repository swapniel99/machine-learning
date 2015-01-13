function d = loader()

persistent d=[];
persistent flag=0;
if flag==0,
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);

[X mu sigma] = featureNormalize(X);

m=length(y);

%X = [ones(m, 1) X];

d=[y,X];
flag=1;
printf("Data loaded\n");
end

end
