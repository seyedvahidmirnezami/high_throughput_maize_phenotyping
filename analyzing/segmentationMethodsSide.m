function [] = segmentationMethodsSide(imageName,cameraName,sideVar,mainPath)

if sideVar==2
    side='left';
else
    side = 'right';
end
mkdir(strcat(pwd,'/',cameraName,'/',side),'segmented')
mkdir(strcat(pwd,'/',cameraName,'/',side),'knn')
mkdir(strcat(pwd,'/',cameraName,'/',side),'knn_adj')
mkdir(strcat(pwd,'/',cameraName,'/',side),'blue')
mkdir(strcat(pwd,'/',cameraName,'/',side),'hsv')
mkdir(strcat(pwd,'/',cameraName,'/',side),'otsu')
mkdir(strcat(pwd,'/',cameraName,'/',side),'segmented2')
mkdir(strcat(pwd,'/',cameraName,'/',side),'matFiles')


catchI=[];
parfor picName=1:size(imageName,1)
    
    if imageName{picName,4}==1
        original = imread(strcat(mainPath,cameraName,'/',imageName{picName,1}));
        try
            if imageName{picName,sideVar}==1
                
                resized = imread(strcat(pwd,'/',cameraName,'/',side,'/cropped/',strtok(imageName{picName},'.'),'-',side,'.JPG'));    % Run the detector.
                
                l = resized;
                RGB2=[];
                RGB2 = imadjust(l,[.2 .3 0; .6 .7 1],[]);
                
                knnSegment1_Complete = knnRB(l,2);
                imwrite(knnSegment1_Complete,strcat(pwd,'/',cameraName,'/',side,'/knn/',strtok(imageName{picName},'.'),'_knn_',side,'.png'));
                knnSegment1 = knnSegment1_Complete(1:floor(size(l,1)/2),:,:);
                tempknnSegment1=knnSegment1(:,:,1);
                biggestKnn1 = biggestComponent(tempknnSegment1);
                if (nnz(biggestKnn1)>0)
                    [mainSpikeKnn1] = mainSpikeAllocation(biggestKnn1);
                else
                    mainSpikeKnn1 = zeros(size(biggestKnn1));
                end
                
                knnSegment2_completre = knnRB(RGB2,2);
                imwrite(knnSegment2_completre,strcat(pwd,'/',cameraName,'/',side,'/knn_adj/',strtok(imageName{picName},'.'),'_knn_adj_',side,'.png'));
                knnSegment2 = knnSegment2_completre(1:floor(size(l,1)*3/4),:,:);
                temp=knnSegment2(:,:,1);
                biggestKnn2 = biggestComponent(~temp);
                if (nnz(biggestKnn2)>0)
                    [mainSpikeKnn2] = mainSpikeAllocation(biggestKnn2);
                else
                    mainSpikeKnn2 = zeros(size(biggestKnn2));
                end
                
                out_otsu_complete = otsu(l);
                imwrite(out_otsu_complete,strcat(pwd,'/',cameraName,'/',side,'/otsu/',strtok(imageName{picName},'.'),'_otsu_',side,'.png'));
                out_otsu = out_otsu_complete(1:floor(size(l,1)*3/4),:);
                p=uint8(out_otsu);
                biggestOtsu = biggestComponent(p);
                if (nnz(biggestOtsu)>0)
                    [mainSpikeOtsu] = mainSpikeAllocation(biggestOtsu);
                else
                    mainSpikeOtsu = zeros(size(biggestOtsu));
                end
                
                
                HSVSegment1_complete = HSVSegment(l);
                imwrite(HSVSegment1_complete,strcat(pwd,'/',cameraName,'/',side,'/hsv/',strtok(imageName{picName},'.'),'_hsv_',side,'.png'));
                HSVSegment1 = HSVSegment1_complete(1:floor(size(l,1)*3/4),:,:);
                tempHSVSegment1=HSVSegment1(:,:,1);
                biggestHSV = biggestComponent(tempHSVSegment1);
                if (nnz(biggestHSV)>0)
                    [mainSpikeHSV1] = mainSpikeAllocation(biggestHSV);
                else
                    mainSpikeHSV1 = zeros(size(biggestHSV));
                end
                
                bluechannelSegment_complete = bluechannel(l);
                imwrite(bluechannelSegment_complete,strcat(pwd,'/',cameraName,'/',side,'/blue/',strtok(imageName{picName},'.'),'_blue_',side,'.png'));
                bluechannelSegment = bluechannelSegment_complete(1:floor(size(l,1)*3/4),:,:);
                tempbluechannelSegment=bluechannelSegment(:,:,1);
                biggesttempbluechannelSegment = biggestComponent(tempbluechannelSegment);
                if (nnz(biggesttempbluechannelSegment)>0)
                    [mainSpikeBlue] = mainSpikeAllocation(biggesttempbluechannelSegment);
                else
                    mainSpikeBlue = zeros(size(biggesttempbluechannelSegment));
                end
                
                
                close all
                h=figure;
                set(h, 'Visible', 'off');
                subplot(2,3,1)
                imshow(l)
                
                
                subplot(2,3,2)
                imshow(mainSpikeKnn1)
                if (nnz(mainSpikeKnn1)>0)
                    [mainPathMainSpikeKnn1,widthDistMainSpikeKnn1,tempMainSpikeKnn1,DistancePlotMainSpikeKnn1,leftBoundaryPlot,rightBoundaryPlot,tempStrPlot]=findWidth(mainSpikeKnn1);
                    title(strcat(imageName{picName,1},'knn1-',tempStrPlot))
                    
                    hold on
                    scatter(mainPathMainSpikeKnn1(:,2),mainPathMainSpikeKnn1(:,1),'*g','LineWidth',2);
                    for t=1:size(DistancePlotMainSpikeKnn1,1)
                        line ([DistancePlotMainSpikeKnn1(t,2),DistancePlotMainSpikeKnn1(t,4),DistancePlotMainSpikeKnn1(t,6)] ,[DistancePlotMainSpikeKnn1(t,1),DistancePlotMainSpikeKnn1(t,3),DistancePlotMainSpikeKnn1(t,5)]);
                    end
                    %                 scatter(rightBoundaryPlot(:,2),rightBoundaryPlot(:,1),'*b','LineWidth',2);
                    %                 scatter(leftBoundaryPlot(:,2),leftBoundaryPlot(:,1),'*r','LineWidth',2);
                else
                    tempStrPlot='No';
                    title(strcat(imageName{picName,1},'knn1-',tempStrPlot))
                    mainPathMainSpikeKnn1 = [];
                    widthDistMainSpikeKnn1=[];
                    tempMainSpikeKnn1=0;
                end
                
                
                subplot(2,3,3)
                imshow(mainSpikeKnn2)
                if (nnz(mainSpikeKnn2)>0)
                    [mainPathMainSpikeKnn2,widthDistMainSpikeKnn2,tempMainSpikeKnn2,DistancePlotMainSpikeKnn2,leftBoundaryPlot,rightBoundaryPlot,tempStrPlot]=findWidth(mainSpikeKnn2);
                    title(strcat('knn2-',tempStrPlot))
                    
                    hold on
                    scatter(mainPathMainSpikeKnn2(:,2),mainPathMainSpikeKnn2(:,1),'*g','LineWidth',2);
                    for t=1:size(DistancePlotMainSpikeKnn2,1)
                        line ([DistancePlotMainSpikeKnn2(t,2),DistancePlotMainSpikeKnn2(t,4),DistancePlotMainSpikeKnn2(t,6)] ,[DistancePlotMainSpikeKnn2(t,1),DistancePlotMainSpikeKnn2(t,3),DistancePlotMainSpikeKnn2(t,5)]);
                    end
                    %                 scatter(rightBoundaryPlot(:,2),rightBoundaryPlot(:,1),'*b','LineWidth',2);
                    %                 scatter(leftBoundaryPlot(:,2),leftBoundaryPlot(:,1),'*r','LineWidth',2);
                else
                    
                    
                    tempStrPlot='No';
                    title(strcat('knn2-',tempStrPlot))
                    mainPathMainSpikeKnn2 = [];
                    widthDistMainSpikeKnn2=[];
                    tempMainSpikeKnn2=0;
                end
                
                subplot(2,3,4)
                imshow(mainSpikeOtsu)
                if (nnz(mainSpikeOtsu)>0)
                    [mainPathMainSpikeotsu,widthDistMainSpikeotsu,tempMainSpikeotsu,DistancePlotMainSpikeotsu,leftBoundaryPlot,rightBoundaryPlot,tempStrPlot]=findWidth(mainSpikeOtsu);
                    title(strcat('Otsu-',tempStrPlot))
                    
                    hold on
                    scatter(mainPathMainSpikeotsu(:,2),mainPathMainSpikeotsu(:,1),'*g','LineWidth',2);
                    for t=1:size(DistancePlotMainSpikeotsu,1)
                        line ([DistancePlotMainSpikeotsu(t,2),DistancePlotMainSpikeotsu(t,4),DistancePlotMainSpikeotsu(t,6)] ,[DistancePlotMainSpikeotsu(t,1),DistancePlotMainSpikeotsu(t,3),DistancePlotMainSpikeotsu(t,5)]);
                    end
                    %                 scatter(rightBoundaryPlot(:,2),rightBoundaryPlot(:,1),'*b','LineWidth',2);
                    %                 scatter(leftBoundaryPlot(:,2),leftBoundaryPlot(:,1),'*r','LineWidth',2);
                else
                    tempStrPlot='No';
                    title(strcat('Otsu-',tempStrPlot))
                    mainPathMainSpikeotsu = [];
                    widthDistMainSpikeotsu=[];
                    tempMainSpikeotsu=0;
                    
                end
                
                subplot(2,3,5)
                imshow(mainSpikeHSV1)
                if (nnz(mainSpikeHSV1)>0)
                    [mainPathMainSpikeHSV1,widthDistMainSpikeHSV1,tempMainSpikeHSV1,DistancePlotMainSpikeHSV1,leftBoundaryPlot,rightBoundaryPlot,tempStrPlot]=findWidth(mainSpikeHSV1);
                    title(strcat('HSV-',tempStrPlot))
                    
                    hold on
                    scatter(mainPathMainSpikeHSV1(:,2),mainPathMainSpikeHSV1(:,1),'*g','LineWidth',2);
                    for t=1:size(DistancePlotMainSpikeHSV1,1)
                        line ([DistancePlotMainSpikeHSV1(t,2),DistancePlotMainSpikeHSV1(t,4),DistancePlotMainSpikeHSV1(t,6)] ,[DistancePlotMainSpikeHSV1(t,1),DistancePlotMainSpikeHSV1(t,3),DistancePlotMainSpikeHSV1(t,5)]);
                    end
                    %                 scatter(rightBoundaryPlot(:,2),rightBoundaryPlot(:,1),'*b','LineWidth',2);
                    %                 scatter(leftBoundaryPlot(:,2),leftBoundaryPlot(:,1),'*r','LineWidth',2);
                else
                    tempStrPlot='No';
                    title(strcat('HSV-',tempStrPlot))
                    
                    mainPathMainSpikeHSV1 = [];
                    widthDistMainSpikeHSV1=[];
                    tempMainSpikeHSV1=0;
                    
                end
                
                subplot(2,3,6)
                imshow(mainSpikeBlue)
                if (nnz(mainSpikeBlue)>0)
                    [mainPathMainSpikeBlue,widthDistMainSpikeBlue,tempMainSpikeBlue,DistancePlotMainSpikeBlue,leftBoundaryPlot,rightBoundaryPlot,tempStrPlot]=findWidth(mainSpikeBlue);
                    title(strcat('Blue-',tempStrPlot))
                    
                    hold on
                    scatter(mainPathMainSpikeBlue(:,2),mainPathMainSpikeBlue(:,1),'*g','LineWidth',2);
                    for t=1:size(DistancePlotMainSpikeBlue,1)
                        line ([DistancePlotMainSpikeBlue(t,2),DistancePlotMainSpikeBlue(t,4),DistancePlotMainSpikeBlue(t,6)] ,[DistancePlotMainSpikeBlue(t,1),DistancePlotMainSpikeBlue(t,3),DistancePlotMainSpikeBlue(t,5)]);
                    end
                    %                 scatter(rightBoundaryPlot(:,2),rightBoundaryPlot(:,1),'*b','LineWidth',2);
                    %                 scatter(leftBoundaryPlot(:,2),leftBoundaryPlot(:,1),'*r','LineWidth',2);
                else
                    tempStrPlot='No';
                    title(strcat('Blue-',tempStrPlot))
                    
                    
                    mainPathMainSpikeBlue = [];
                    widthDistMainSpikeBlue=[];
                    tempMainSpikeBlue=0;
                    
                end
                
                
                DistancePlotAll{picName,1}{1,1}=DistancePlotMainSpikeKnn1;
                DistancePlotAll{picName,1}{1,2}=DistancePlotMainSpikeKnn2;
                DistancePlotAll{picName,1}{1,3}=DistancePlotMainSpikeotsu;
                DistancePlotAll{picName,1}{1,4}=DistancePlotMainSpikeHSV1;
                DistancePlotAll{picName,1}{1,5}=DistancePlotMainSpikeBlue;
                
                
                tempAll{picName,1}{1,1}=tempMainSpikeKnn1;
                tempAll{picName,1}{1,2}=tempMainSpikeKnn2;
                tempAll{picName,1}{1,3}=tempMainSpikeotsu;
                tempAll{picName,1}{1,4}=tempMainSpikeHSV1;
                tempAll{picName,1}{1,5}=tempMainSpikeBlue;
                
                
                widthAll{picName,1}{2,1}=widthDistMainSpikeKnn1;
                widthAll{picName,1}{2,2}=widthDistMainSpikeKnn2;
                widthAll{picName,1}{2,3}=widthDistMainSpikeotsu;
                widthAll{picName,1}{2,4}=widthDistMainSpikeHSV1;
                widthAll{picName,1}{2,5}=widthDistMainSpikeBlue;
                
                
                mainPathAll{picName,1}{1,1}=mainPathMainSpikeKnn1;
                mainPathAll{picName,1}{1,2}=mainPathMainSpikeKnn2;
                mainPathAll{picName,1}{1,3}=mainPathMainSpikeotsu;
                mainPathAll{picName,1}{1,4}=mainPathMainSpikeHSV1;
                mainPathAll{picName,1}{1,5}=mainPathMainSpikeBlue;
                
                
                segmentedImage{picName,1}{1,1}=mainSpikeKnn1;
                segmentedImage{picName,1}{1,2}=mainSpikeKnn2;
                segmentedImage{picName,1}{1,3}=mainSpikeOtsu;
                segmentedImage{picName,1}{1,4}=mainSpikeHSV1;
                segmentedImage{picName,1}{1,5}=mainSpikeBlue;
                
                
                
                picName
            else
                tempAll{picName,1}{1,1}=-1;
                tempAll{picName,1}{1,2}=-1;
                tempAll{picName,1}{1,3}=-1;
                tempAll{picName,1}{1,4}=-1;
                tempAll{picName,1}{1,5}=-1;
                
                close all
                h=figure;
                set(h, 'Visible', 'off');
                imshow(original)
                title('noLeftAndRight')
                
            end
            
        catch
            tempAll{picName,1}{1,1}=-1;
            tempAll{picName,1}{1,2}=-1;
            tempAll{picName,1}{1,3}=-1;
            tempAll{picName,1}{1,4}=-1;
            tempAll{picName,1}{1,5}=-1;
            
            catchI=[catchI;picName]
            
            
            close all
            h=figure;
            set(h, 'Visible', 'off');
            
            subplot(1,2,1)
            imshow(original)
            title('catch')
            % subplot(1,2,2)
            % imshow(resized)
            
            
        end
        
        
    else
        close all
        h=figure;
        set(h, 'Visible', 'off');
        
        subplot(1,2,1)
        title('noImage')
        tempAll{picName,1}{1,1}=-1;
        tempAll{picName,1}{1,2}=-1;
        tempAll{picName,1}{1,3}=-1;
        tempAll{picName,1}{1,4}=-1;
        tempAll{picName,1}{1,5}=-1;
    end
    saveas(h,strcat(pwd,'/',cameraName,'/',side,'/segmented/',strtok(imageName{picName,1},'.'),'-',side,'.jpg'));
end
save(strcat(pwd,'/',cameraName,'/',side,'/matFiles/segmented-',cameraName,'-',side,'.mat'),'-v7.3');
clearvars -except mainPath tempAll segmentedImage mainPathAll DistancePlotAll widthAll imageName cameraName side
no_fig=0;
primaryResult=cell(length(imageName),6);
maxSizeSpike = -1;
for i=1:size(imageName,1)
    top_part= 1;
    no_fig=0;
    maxSize = -1;
    for j=1:5
        if tempAll{i,1}{1,j}==1
            if mainPathAll{i,1}{1,j}(1,1)>top_part+50
                if top_part==1
                    top_part = mainPathAll{i,1}{1,j}(1,1);
                    
                    if size(mainPathAll{i,1}{1,j},1)>maxSize
                        maxSize= size(mainPathAll{i,1}{1,j},1);
                        index=j;
                    end
                end
            else
                if size(mainPathAll{i,1}{1,j},1)>maxSize
                    maxSize= size(mainPathAll{i,1}{1,j},1);
                    index=j;
                end
            end
            
        end
    end
    if maxSize > maxSizeSpike
        maxSizeSpike = maxSize;
    end
    
    if maxSize==-1
        primaryResult{i,1}=imageName{i,1};
        primaryResult{i,6}=1;
        no_fig=1;
    else
        if ~isempty(find(segmentedImage{i,1}{1,index}(1,:)==1))
            primaryResult{i,1}=imageName{i,1};
            primaryResult{i,6}=1;
            no_fig=1;
        else
            
            primaryResult{i,6}=0;
            primaryResult{i,1}=imageName{i,1};
            primaryResult{i,2}=index;
            primaryResult{i,3}=mainPathAll{i,1}{1,index};
            primaryResult{i,4}=segmentedImage{i,1}{1,index};
            primaryResult{i,5}=widthAll{i,1}{2,index};
            close all
            h=figure;
            set(h, 'Visible', 'on');
            subplot(1,2,1)
            resized = imread(strcat(pwd,'/',cameraName,'/',side,'/cropped/',strtok(imageName{i,1},'.'),'-',side,'.JPG'));    % Run the detector.
            
            
            imshow(resized)
            
            
            
            hold on
            %             scatter(mainPathAll{i,1}{1,index}(:,2),mainPathAll{i,1}{1,index}(:,1),'*g','LineWidth',2);
            for t=1:200:size(DistancePlotAll{i,1}{1,index},1)
                line ([DistancePlotAll{i,1}{1,index}(t,2),DistancePlotAll{i,1}{1,index}(t,4),...
                    DistancePlotAll{i,1}{1,index}(t,6)] ,[DistancePlotAll{i,1}{1,index}(t,1),...
                    DistancePlotAll{i,1}{1,index}(t,3),DistancePlotAll{i,1}{1,index}(t,5)],'Color','red','LineWidth',3);
            end
            t= size(DistancePlotAll{i,1}{1,index},1);
            line ([DistancePlotAll{i,1}{1,index}(t,2),DistancePlotAll{i,1}{1,index}(t,4),...
                DistancePlotAll{i,1}{1,index}(t,6)] ,[DistancePlotAll{i,1}{1,index}(t,1),...
                DistancePlotAll{i,1}{1,index}(t,3),DistancePlotAll{i,1}{1,index}(t,5)],'Color','red','LineWidth',3);
            
            %
            %             row_y_sorted=sort(unique(mainPathAll{i,1}{1,index}(:,1)));
            %             for i_row=1:100:size(row_y_sorted,1)
            %                 hline = refline([0 row_y_sorted(i_row,1)]);
            %             end
            %             hline = refline([0 size(segmentedImage{i,1}{1,index},1)]);
            subplot(1,2,2)
            imshow(segmentedImage{i,1}{1,index})
            hold on
            %             scatter(mainPathAll{i,1}{1,index}(:,2),mainPathAll{i,1}{1,index}(:,1),'*g','LineWidth',0.05);
            for t=1:200:size(DistancePlotAll{i,1}{1,index},1)
                line ([DistancePlotAll{i,1}{1,index}(t,2),DistancePlotAll{i,1}{1,index}(t,4),...
                    DistancePlotAll{i,1}{1,index}(t,6)] ,[DistancePlotAll{i,1}{1,index}(t,1),...
                    DistancePlotAll{i,1}{1,index}(t,3),DistancePlotAll{i,1}{1,index}(t,5)],'Color','red','LineWidth',3);
            end
            title(imageName{i,1})
            
        end
        
    end
    if no_fig
        %         if exist(strcat(pwd,'/',cameraName,'/',side,'/cropped/',strtok(imageName{i,1},'.'),'-',side,'.JPG'))
        try
            resized = imread(strcat(pwd,'/',cameraName,'/',side,'/cropped/',strtok(imageName{i,1},'.'),'-',side,'.JPG'));    % Run the detector.
        catch
            resized = ones(200,200);
            
        end
        close all
        h=figure;
        set(h, 'Visible', 'off');
        subplot(1,2,1)
        imshow(resized)
        title(imageName{i,1})
    end
    saveas(h,strcat(pwd,'/',cameraName,'/',side,'/segmented2/',strtok(imageName{i,1},'.'),'-',side,'.jpg'));
end

widthFinal = nan(maxSizeSpike+1,size(primaryResult,1));
widthFinal = num2cell(widthFinal);

for i=1:size(primaryResult,1)
    widthFinal{1,i} = primaryResult{i,1};
    if (primaryResult{i,6}==0)
        for j=2:size(primaryResult{i,5},1)
            widthFinal{j,i} = primaryResult{i,5}(j,1);
        end
    end
end

save(strcat(pwd,'/',cameraName,'/',side,'/matFiles/WidthResult-',cameraName,'-',side,'.mat'),'widthFinal','-v7.3');
save(strcat(pwd,'/',cameraName,'/',side,'/matFiles/primaryResult-',cameraName,'-',side,'.mat'),'primaryResult','-v7.3');
save(strcat(pwd,'/',cameraName,'/',side,'/matFiles/widthAll-',cameraName,'-',side,'.mat'),'widthAll','-v7.3');

writetable(cell2table(widthFinal) , strcat(pwd,'/',cameraName,'/',side,'/',cameraName,'_',side,'_width.csv'))
writetable(cell2table(widthFinal) , strcat(pwd,'/',cameraName,'/',side,'/',cameraName,'_',side,'_width.xlsx'))
mkdir(pwd,'tables_csv')
mkdir(pwd,'tables_excel')
writetable(cell2table(widthFinal) , strcat(pwd,'/tables_csv/',cameraName,'_',side,'_width.csv'))
writetable(cell2table(widthFinal) , strcat(pwd,'/tables_excel/',cameraName,'_',side,'_width.xlsx'))
