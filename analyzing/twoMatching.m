function [sideCheck,leftResult,rightResult] = twoMatching(leftImage,rightImage,croppedImages,croppedImagesSide)
leftResult = leftImage;
rightResult = rightImage;

if (~isempty(leftImage) || ~isempty(rightImage))
    
    leftFeature1 = checkMatchFeature(leftImage,croppedImages{1,1});
    leftFeature2 = checkMatchFeature(leftImage,croppedImages{2,1});
    rightFeature1 = checkMatchFeature(rightImage,croppedImages{1,1});
    rightFeature2 = checkMatchFeature(rightImage,croppedImages{2,1});
    
    result = [leftFeature1 11;leftFeature2 12;rightFeature1 21;rightFeature2 22];
    resultSorted = sortrows(result,'descend');
    
    if ~isempty(find(resultSorted(:,1)>0)) & abs(abs(resultSorted(1,1))-abs(resultSorted(2,1)))>10
        if resultSorted(1,2)==11
            leftResult = croppedImages{1,1};
            rightResult = croppedImages{2,1};
            sideCheck = 1;
        elseif resultSorted(1,2)==12
            leftResult = croppedImages{2,1};
            rightResult = croppedImages{1,1};
            sideCheck = 0;
        elseif resultSorted(1,2)==21
            rightResult = croppedImages{1,1};
            leftResult = croppedImages{2,1};
            sideCheck = 0;
        else
            rightResult = croppedImages{2,1};
            leftResult = croppedImages{1,1};
            sideCheck = 1;
        end
    else
            
        for i=1:2
            if croppedImagesSide(i,1)==1
                rightResult = croppedImages{i,1};
                if i==1
                    sideCheck = 0;
                else
                    sideCheck = 1;
                end
            else
                leftResult = croppedImages{i,1};
                if i==1
                    sideCheck = 1;
                else
                    sideCheck = 0;
                end
            end
        end
        
    end
    
    
    
elseif isempty(rightImage) & isempty(leftImage)
    for i=1:2
        if croppedImagesSide(i,1)==1
            rightResult = croppedImages{i,1};
            if i==1
                sideCheck = 0;
            else
                sideCheck = 1;
            end
        else
            leftResult = croppedImages{i,1};
            if i==1
                sideCheck = 1;
            else
                sideCheck = 0;
            end
        end
    end
    
end




end



