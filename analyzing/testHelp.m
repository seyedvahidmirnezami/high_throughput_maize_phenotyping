mainPath='/media/vahid/6A96EF6096EF2AF1/';
load(strcat('/media/vahid/96CC807ACC805701/anthesis/fasterRCNNTassel2016/Matchfeatures/12April','/CAM483/matFiles/cropping-CAM483.mat'),'picNameAll','imageName','cameraName');
tic
sideVar=3;
segmentationMethodsSide(imageName,cameraName,sideVar,mainPath)
disp('done2')
toc
clearvars -except imageName cameraName mainPath
tic
sideVar=3;
segmentationMethodsSide(imageName,cameraName,sideVar,mainPath)
disp('done3')
toc