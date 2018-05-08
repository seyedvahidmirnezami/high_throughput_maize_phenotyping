function [leftResult,rightResult] = oneMatching(leftImage,rightImage,croppedImages,croppedImagesSide)
leftResult = leftImage;
rightResult = rightImage;

leftFeature = checkMatchFeature(leftImage,croppedImages{1,1});
rightFeature = checkMatchFeature(rightImage,croppedImages{1,1});

if (~isempty(leftImage) || ~isempty(rightImage))
    if abs(abs(leftFeature)-abs(rightFeature))>10
        if (leftFeature > rightFeature )
            leftResult = croppedImages{1,1};
            rightResult = [];
            
        elseif (rightFeature > leftFeature)
            rightResult = croppedImages{1,1};
            leftResult = [];
        end
    else
        
        
        if (croppedImagesSide==-1)
            leftResult = croppedImages{1,1};
            rightResult = [];
        else
            rightResult = croppedImages{1,1};
            leftResult = [];
        end
    end
    
elseif (isempty(leftImage) & isempty(rightImage))
    if (croppedImagesSide==-1)
        leftResult = croppedImages{1,1};
        rightResult = [];
    else
        rightResult = croppedImages{1,1};
        leftResult = [];
    end
    
end
end