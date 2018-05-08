mainPath = '/work/baskargroup/vahid/2016Stationary/fasterRCNNTassel2016/rightAndLeft/images/';
load(strcat(pwd,'/CAM556/matFiles/cropping-CAM556.mat'),'picNameAll','imageName','cameraName');
tic
sideVar=2;
segmentationMethodsSide(imageName,cameraName,sideVar,mainPath)
disp('done2')
toc
clearvars -except imageName cameraName mainPath
tic
sideVar=3;
segmentationMethodsSide(imageName,cameraName,sideVar,mainPath)
disp('done3')
toc
