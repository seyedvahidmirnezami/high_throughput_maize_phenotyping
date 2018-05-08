% clearvars -except I1 I2 I3 IMG l l1

% dir=dir('C:\Users\meisu\Desktop\H17-Final-filter\*.jpg');

% I1=imread(dir(1).name);
% I2=imread(dir(8).name);
function resultFeature = checkMatchFeature(I1,I2)

if (isempty(I1) || isempty(I2))
    resultFeature = -1;
else
    
I1=rgb2gray(I1);
I2=rgb2gray(I2);
points1 = detectMinEigenFeatures(I1);
points2 = detectMinEigenFeatures(I2);

[features1,valid_points1] = extractFeatures(I1,points1);
[features2,valid_points2] = extractFeatures(I2,points2);

indexPairs = matchFeatures(features1,features2);
resultFeature = size(indexPairs,1);

end

% detectKAZEFeatures
% detectBRISKFeatures
% detectFASTFeatures
% detectHarrisFeatures
% % detectMSERFeatures