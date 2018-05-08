function HSV= HSVSegment (I)

% clear all
% I=imread('/home/vahid/Desktop/Colton/6inches/6inches.jpg');
% I=I(:,265:end,:);

hue_threshold = [0 0.18]; % Fill this in


% Read the image file
% I = imread(filename);
% I=imread('044-11-20150901_1405.JPG');

% Converts image to hsv format
hsv = rgb2hsv(I);

% Isolate hue channel. 
h = hsv(:,:,1);

% Segment region of interest based on Hue values. Can be modified for other values
ROI = h > hue_threshold(1) & h < hue_threshold(2); % Can remove 2nd threshold if not needed  

% Creates mask of segmented image
mask = repmat(ROI,[1,1,3]);

% Applies mask onto image. 
I(~mask) = 0;
% I(mask) = 0; % Depends on whether you want to keep the mask or remove.

HSV=I;
% imwrite(I,'HSV.jpg');
