function [newbboxesAfterOverLapSorted,croppedImages,croppedImagesSide] = checkOverlap(newbboxes,I)

for t=1:size(newbboxes ,1)
    if (newbboxes(t,2)-500>0)
        topBorder(t,1) = newbboxes(t,2)-500;
        bottomBorder(t,1) = newbboxes(t,4)+500;
    else
        topBorder(t,1)=1;
        bottomBorder(t,1)=newbboxes(t,4)+newbboxes(t,2);
    end
end

newbboxes = [newbboxes(:,1) topBorder(:,1) newbboxes(:,3) bottomBorder(:,1)];

if size(newbboxes,1)==1
    %     newbboxesAfterOverLapSorted = [newbboxes(:,1) repmat(1,size(newbboxes,1),1) newbboxes(:,3) newbboxes(:,4)+newbboxes(:,2)];
    
    newbboxesAfterOverLapSorted = [newbboxes(:,1) newbboxes(:,2) newbboxes(:,3) newbboxes(:,4)];
    croppedImages{1,1} = imcrop(I,newbboxesAfterOverLapSorted);
    if (newbboxesAfterOverLapSorted(1,1)+newbboxesAfterOverLapSorted(1,3)) > 3*size(I,2)/5
        croppedImagesSide(1,1) = 1;
    else
        croppedImagesSide(1,1) = -1;
    end
    
else
    overlap=zeros(size(newbboxes,1),size(newbboxes,1));
    
    for i=1:size(newbboxes,1)-1
        for j=i+1:size(newbboxes,1)
            right1 = newbboxes(i,1) + newbboxes(i,3);
            right2 = newbboxes(j,1) + newbboxes(j,3);
            left1 = newbboxes(i,1);
            left2 = newbboxes(j,1);
            
            
            if ((right1<right2 & left2<left1) || (right2<right1 & left1<left2))
                overlap(i,j)=j;
            elseif abs(right1-right2)<3*(min(newbboxes(j,3),newbboxes(i,3)))/4
                overlap(i,j)=j;
            elseif abs(left1-left2)<3*(min(newbboxes(j,3),newbboxes(i,3)))/4
                overlap(i,j)=j;
            end
            
            
            
        end
    end
    newbboxesAfterOverLap = [];
    if nnz(overlap)>0
        for i=1:size(newbboxes,1)
            tempRight = newbboxes(i,1) + newbboxes(i,3);
            tempLeft = newbboxes(i,1);
            tempTop = newbboxes(i,2) ;
            tempBottom = newbboxes(i,2) + newbboxes(i,4);
            if nnz(overlap(i,:))>0
                for j=1:size(newbboxes,1)
                    if nnz(overlap(i,j))~=0
                        
                        tempRight = max(tempRight , newbboxes(j,1) + newbboxes(j,3));
                        tempLeft = min(tempLeft , newbboxes(j,1));
                        tempTop = min(tempTop , newbboxes(j,2));
                        tempBottom = max(tempBottom, newbboxes(j,2) + newbboxes(j,4));
                    end
                    
                end
                newbboxesAfterOverLap=[newbboxesAfterOverLap;(tempRight-tempLeft)*(tempBottom-tempTop) tempLeft tempTop tempRight-tempLeft tempBottom-tempTop];
            end
        end
    else
        newbboxesAfterOverLap=[newbboxesAfterOverLap;newbboxes(:,3).*newbboxes(:,4) newbboxes(:,1) newbboxes(:,2) newbboxes(:,3) newbboxes(:,4)];
    end
    newbboxesAfterOverLapSorted = sortrows(newbboxesAfterOverLap,'descend');
    
    for i=1:size(newbboxesAfterOverLapSorted,1)
        if i<=2
            croppedImages{i,1} = imcrop(I,newbboxesAfterOverLapSorted(i,2:5));
            if (newbboxesAfterOverLapSorted(1,2)+newbboxesAfterOverLapSorted(1,4)) > 3*size(I,2)/5
                croppedImagesSide(i,1) = 1;
            else
                croppedImagesSide(i,1) = -1;
            end
        end
    end
    
    if i~=1
        if newbboxesAfterOverLapSorted(1,2)+newbboxesAfterOverLapSorted(1,4) > newbboxesAfterOverLapSorted(2,2)+newbboxesAfterOverLapSorted(2,4)
            croppedImages{1,1} = imcrop(I,newbboxesAfterOverLapSorted(1,2:5));
            croppedImages{2,1} = imcrop(I,newbboxesAfterOverLapSorted(2,2:5));
            croppedImagesSide(1,1) = 1;
            croppedImagesSide(2,1) = -1;
        else
            croppedImages{2,1} = imcrop(I,newbboxesAfterOverLapSorted(1,2:5));
            
            croppedImages{1,1} = imcrop(I,newbboxesAfterOverLapSorted(2,2:5));
            croppedImagesSide(1,1) = 1;
            croppedImagesSide(2,1) = -1;
            temp=[newbboxesAfterOverLapSorted(2,:); newbboxesAfterOverLapSorted(1,:)];
            newbboxesAfterOverLapSorted = temp;
        end

    end
    
end

