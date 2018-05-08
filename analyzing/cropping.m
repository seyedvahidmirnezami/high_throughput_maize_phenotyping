function []=cropping(num)
% num=483;
% mainPath = '/media/vahid/6A96EF6096EF2AF1/';
mainPath = '/work/alurugroup/eddyyeh/AnthesisProgression/2016_OrganizedPhotos/';
% mainPath = '/work/baskargroup/vahid/2016Stationary/fasterRCNNTassel2016/rightAndLeft/images/';
num
cameraName=strcat('CAM',num2str(num));
strcat(mainPath,cameraName)
if exist(strcat(mainPath,cameraName))
num    
load('fasterRcnnTasselDetector.mat');
mkdir(pwd,strcat(cameraName))
mkdir(strcat(pwd,'/',cameraName),'left')
mkdir(strcat(pwd,'/',cameraName),'right')

mkdir(strcat(pwd,'/',cameraName),'boxes')
mkdir(strcat(pwd,'/',cameraName,'/left'),'cropped')
mkdir(strcat(pwd,'/',cameraName,'/right'),'cropped')
mkdir(strcat(pwd,'/',cameraName),'matFiles')




ss=dir(strcat(mainPath,cameraName,'/*.JPG'));
picNameAll=[];
tempLeft = [];
tempRight = [];

for picName=1:length(ss)
    
    bad=0;
    picName
    tempOne=0;
    tempTwo=0;
    try
        clear I;
        I = imread(strcat(mainPath,cameraName,'/',ss(picName).name));
    catch
        bad=1;
    end
    try
        
        imageName{picName,1} = ss(picName).name;
        
        tempSide='no';
        % Run the detector.
        resized=imresize(I,0.1);
        
        imageName{picName,4} = 1;
        [bboxes, scores] = detect(detector, resized);
        
        
        if ~isempty(bboxes)
            boxAndI = insertObjectAnnotation(imresize(I,0.1), 'rectangle', bboxes, scores);
            
            [a,b] = sort(scores,'descend');
            if (size(b,1)==1)
                newbboxes = bboxes(b(1,:)',:)*10;
            else
                newbboxes = bboxes(b(1:2,:)',:)*10;
            end
            
            [newbboxesAfterOverLapSorted,croppedImages,croppedImagesSide] = checkOverlap(newbboxes,I);
            
            
            if (size(croppedImages,1)==1)
                
                [tempLeft,tempRight] = oneMatching(tempLeft,tempRight,croppedImages,croppedImagesSide);
                tempOne=1;
                
            elseif (size(croppedImages,1)==2)
                
                [sideCheck,tempLeft,tempRight] = twoMatching(tempLeft,tempRight,croppedImages,croppedImagesSide);
                tempTwo=1;
                
            end
            
            if tempOne
                if ~isempty(tempRight)
                    imwrite(tempRight,strcat(pwd,'/',cameraName,'/right/cropped/', strtok(ss(picName).name,'.'),'-right.JPG'));
                    imageName{picName,3} = 1;
                    if size(newbboxesAfterOverLapSorted,2)<=4
                        boxAndI = insertObjectAnnotation(boxAndI, 'rectangle', newbboxesAfterOverLapSorted(1,1:4)/10, 'Right','Color',{'blue'},'FontSize',10);
                    end
                    if size(newbboxesAfterOverLapSorted,2)==5
                        boxAndI = insertObjectAnnotation(boxAndI, 'rectangle', newbboxesAfterOverLapSorted(1,2:5)/10, 'Right','Color',{'blue'},'FontSize',10);
                    end
                end
                if ~isempty(tempLeft)
                    imwrite(tempLeft,strcat(pwd,'/',cameraName,'/left/cropped/', strtok(ss(picName).name,'.'),'-left.JPG'));
                    imageName{picName,2} = 1;
                    if size(newbboxesAfterOverLapSorted,2)<=4
                        boxAndI = insertObjectAnnotation(boxAndI, 'rectangle', newbboxesAfterOverLapSorted(1,1:4)/10, 'Left','Color',{'red'},'FontSize',10);
                    end
                    if size(newbboxesAfterOverLapSorted,2)==5
                        boxAndI = insertObjectAnnotation(boxAndI, 'rectangle', newbboxesAfterOverLapSorted(1,2:5)/10, 'Left','Color',{'red'},'FontSize',10);
                    end
                end
            end
            if tempTwo
                if ~isempty(tempRight)
                    imwrite(tempRight,strcat(pwd,'/',cameraName,'/right/cropped/', strtok(ss(picName).name,'.'),'-right.JPG'));
                    imageName{picName,3} = 1;
                    
                    if sideCheck
                        boxAndI = insertObjectAnnotation(boxAndI, 'rectangle', newbboxesAfterOverLapSorted(2,2:5)/10, 'Right','Color',{'blue'},'FontSize',10);
                    else
                        boxAndI = insertObjectAnnotation(boxAndI, 'rectangle', newbboxesAfterOverLapSorted(1,2:5)/10, 'Right','Color',{'blue'},'FontSize',10);
                    end
                end
                if ~isempty(tempLeft)
                    imwrite(tempLeft,strcat(pwd,'/',cameraName,'/left/cropped/', strtok(ss(picName).name,'.'),'-left.JPG'));
                    imageName{picName,2} = 1;
                    if sideCheck
                        boxAndI = insertObjectAnnotation(boxAndI, 'rectangle', newbboxesAfterOverLapSorted(1,2:5)/10, 'Left','Color',{'red'},'FontSize',10);
                    else
                        boxAndI = insertObjectAnnotation(boxAndI, 'rectangle', newbboxesAfterOverLapSorted(2,2:5)/10, 'Left','Color',{'red'},'FontSize',10);
                    end
                end
            end
            imwrite(boxAndI,strcat(pwd,'/',cameraName,'/boxes/',ss(picName).name));
        end
        
        
    catch
        if bad==0
            picNameAll=[picNameAll;picName];
            imageName{picName,1} = ss(picName).name;
            imageName{picName,2} = 0;
            imageName{picName,3} = 0;
            imageName{picName,4} = 1;
        else
            picNameAll=[picNameAll;picName];
            imageName{picName,1} = ss(picName).name;
            imageName{picName,2} = 0;
            imageName{picName,3} = 0;
            imageName{picName,4} = -1; % image cannot be read at all
            
        end
        
    end
end

save(strcat(pwd,'/',cameraName,'/','cropping-',cameraName,'.mat'),'picNameAll','imageName','cameraName');
clearvars -except mainPath sideVar cameraName imageName
tic
sideVar=2;
segmentationMethodsSide(imageName,cameraName,sideVar,mainPath)
leftToc=toc
save(strcat(pwd,'/',cameraName,'/left/','timeLeft-',cameraName,'.mat'),'leftToc');
clearvars -except mainPath sideVar cameraName imageName
tic
sideVar=3;
segmentationMethodsSide(imageName,cameraName,sideVar,mainPath)
righToc=toc

save(strcat(pwd,'/',cameraName,'/right/','timeRight-',cameraName,'.mat'),'righToc');
end
