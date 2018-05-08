cameraName= 'CAM483';
side = 'right';
load(strcat(pwd,'/',cameraName,'/matFiles/segmented-',cameraName,'-',side,'.mat'));
clearvars -except tempAll segmentedImage mainPathAll DistancePlotAll widthAll imageName cameraName side
no_fig=0;
primaryResult=cell(length(imageName),6);
maxSizeSpike = -1;
for i=1:size(imageName,1)
i
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
            resized = imread(strcat(pwd,'/',cameraName,'/cropped/',side,'/',strtok(imageName{i,1},'.'),'-',side,'.JPG'));    % Run the detector.
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
        if exist(strcat(pwd,'/',cameraName,'/cropped/',side,'/',strtok(imageName{i,1},'.'),'-',side,'.JPG'))
            resized = imread(strcat(pwd,'/',cameraName,'/cropped/',side,'/',strtok(imageName{i,1},'.'),'-',side,'.JPG'));    % Run the detector.
        else
            resized = imread(strcat(pwd,'/images/',cameraName,'/',imageName{i,1}));    % Run the detector.
            
        end
        close all
        h=figure;
        set(h, 'Visible', 'off');
        subplot(1,2,1)
        imshow(resized)
        title(imageName{i,1})
    end
    saveas(h,strcat(pwd,'/',cameraName,'/segmented2/',side,'/',strtok(imageName{i,1},'.'),'-',side,'.jpg'));
end

widthFinal = nan(maxSizeSpike,size(primaryResult,1));
for i=1:size(primaryResult,1)
    if (primaryResult{i,6}==0)
        for j=1:size(primaryResult{i,5},1)
            widthFinal(j,i) = primaryResult{i,5}(j,1);
        end
    end
end

save(strcat(pwd,'/',cameraName,'/matFiles/WidthResult-',cameraName,'-',side,'.mat'),'widthFinal','-v7.3');
save(strcat(pwd,'/',cameraName,'/matFiles/primaryResult-',cameraName,'-',side,'.mat'),'primaryResult','-v7.3');
save(strcat(pwd,'/',cameraName,'/matFiles/widthAll-',cameraName,'-',side,'.mat'),'widthAll','-v7.3');
