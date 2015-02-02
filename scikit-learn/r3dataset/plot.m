
data = load('trainr3.csv');
X = data(:,1);
Y = data(:,2);
Z = data(:,3);

data0 = load('testr30.csv');
X0 = data0(:,1);
Y0 = data0(:,2);
Z0 = data0(:,3);

data1 = load('testr31.csv');
X1 = data1(:,1);
Y1 = data1(:,2);
Z1 = data1(:,3);

figure;
scatter3(X,Y,Z,[],'r');
hold on;
scatter3(X0,Y0,Z0,[],'b');
scatter3(X1,Y1,Z1,[],'g');

