width=[];
a=[];
counter=1;
s = dir(strcat('/media/vahid/96CC807ACC805701/anthesis/fasterRCNNTassel2016/',...
                'Matchfeatures/12April/CAM483/segmented/left/*.jpg'));
for i=1:size(widthFinal,2)
%     i
%     counter
    if ~isnan(widthFinal(200,i))
        if nnz(widthFinal(1:200,i)<100)==200
            x=[i;widthFinal(1:200,i)];
            width=[width x];
            copyfile(strcat('/media/vahid/96CC807ACC805701/anthesis/fasterRCNNTassel2016/',...
                'Matchfeatures/12April/CAM483/segmented2/left/',s(i).name),...
                strcat('/media/vahid/96CC807ACC805701/anthesis/fasterRCNNTassel2016/Matchfeatures/',...
                '12April/CAM483/segmented2/left-subset/',num2str(counter),'.jpg'));
            a=[a mean(x(1:50,:))];
            counter=counter+1;
        end
    end
end