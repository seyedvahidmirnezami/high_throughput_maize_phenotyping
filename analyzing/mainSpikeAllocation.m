function [mainSpike] = mainSpikeAllocation(binaryImage)

skeBWbinaryImage = bwmorph(binaryImage,'thin',inf);
[longpath]=LongestPath(skeBWbinaryImage);
longpathSorted = sortrows(longpath,'ascend');
Iblur = imgaussfilt(binaryImage,5);
skeBWIblur = bwmorph(Iblur,'thin',inf);
bpsMat=bwmorph(skeBWIblur,'branchpoints');
[brpsRow bpsCol]=find(bpsMat);
bps=[brpsRow bpsCol];
bpsAll=[bps; bps(:,1) bps(:,2)+1;bps(:,1) bps(:,2)-1];
[longpathBlur]=LongestPath(skeBWIblur);
intersection = intersect(longpathBlur,bpsAll,'rows');
intersectionSorted = sortrows(intersection,'ascend');
if (~isempty(intersectionSorted))
    topPoint = intersectionSorted(1,:);
    mainSpikeTemp = biggestComponent(binaryImage(1:topPoint,:));
else
    mainSpikeTemp = biggestComponent(binaryImage(1:end,:));
end

temp=0;
for enter=size(mainSpikeTemp,1):-1:1
    [~, numberOfObject] = bwlabel(mainSpikeTemp(enter,:));
    if numberOfObject>1
        mainSpike = biggestComponent(mainSpikeTemp(1:enter,:));
        if nnz(mainSpike) ~= nnz(mainSpikeTemp(1:enter,:))
            temp=1;
            break;
        end
        
    end
end
if temp==0
    mainSpike=zeros(size(binaryImage));
end
end