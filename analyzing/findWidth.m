function [mainPath,widthDist,temp,Distance,leftBoundary,rightBoundary,tempStr] = findWidth(binary)

count=1;
for j=1:size(binary,1)
    checkB = binary(j,:);
    [r c] = find(checkB);
    if ~isempty(c)
        leftBoundary(count,1:2) = [j min(c)];
        rightBoundary(count,1:2) = [j max(c)];
        binary(j,min(c):max(c))=1;
        count = count + 1;
    end
end

skeBW = bwmorph(binary,'thin',inf);
skeBW(1,:)=0;
skeBW(end,:)=0;
skeBW(:,1)=0;
skeBW(:,end)=0;
skeBW = biggestComponent(skeBW);
[DistMat]=FourthFindGraph(skeBW);
nn=bwmorph(skeBW,'endpoints');
[vv ww]=find(nn);
Endpoints = [vv ww];
sortedEndpoints = sortrows(Endpoints);
[yskeBW xskeBW]=find(skeBW);
Istart=find(xskeBW==sortedEndpoints(end,2) & yskeBW==sortedEndpoints(end,1));
distCheck=-1;
for endIter=1:size(Endpoints,1)
    
    linearIndEnd=find(xskeBW==Endpoints(endIter,2) & yskeBW==Endpoints(endIter,1));
    [dist, path, pred]=graphshortestpath(DistMat,Istart,linearIndEnd);
    finalPath = [yskeBW(path) xskeBW(path)];
    if (size(unique(finalPath,'rows'),1)>distCheck)
        distCheck=size(finalPath,1);
        mainLinearIndEnd = linearIndEnd;
        mainLinearIndStart = Istart;
        mainPath=finalPath;
    end
    
end
mainPath=sortrows(unique(mainPath,'rows'));
smoothPath = smoothdata(mainPath,'sgolay','SmoothingFactor',1);

count = 0;
Distance = [];
if (size(smoothPath,1)>2)
for i=2:size(smoothPath,1)-1
    
    count = count + 1;
    slope = (smoothPath(i-1,1) - smoothPath(i+1,1)) / ...
        (smoothPath(i-1,2) - smoothPath(i+1,2));
    m = -1/slope;
    tempData = -m*smoothPath(i,2)+smoothPath(i,1);
    resultRight = abs(m*rightBoundary(:,2)-rightBoundary(:,1) + tempData);
    [numRight,idxRight] = min(resultRight);
    resultLeft = abs(m*leftBoundary(:,2)-leftBoundary(:,1) + tempData);
    [numLeft,idxLeft] = min(resultLeft);
    Distance= [Distance;[leftBoundary(idxLeft,:) smoothPath(i,:) rightBoundary(idxRight,:)]];
    
    
    
end
w=[];
X = Distance(:,1:2);
Y = Distance(:,5:6);
w = diag(pdist2(X,Y));
widthDist=w(:,1);
else
    widthDist=[];
    w=[];
end
if ~isempty(w)
    if max(w(:,1))<200
        temp=1;
        tempStr = 'Yes'
    else
         temp=0;
    tempStr = 'No'
    end
    
else
    temp=0;
    tempStr = 'No'
    
    
end


