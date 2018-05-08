function out_otsu = otsu(I)
% I=imread('/home/vahid/Desktop/Colton/24inches/24inches.jpg');
% I=I(:,265:end,:);
I_gray = rgb2gray(I);
level = graythresh(I_gray);
out_otsu = im2bw(I_gray,level);
[r1 c1]=size(out_otsu);
if (nnz(out_otsu)>(r1*c1)/2)
    out_otsu=imcomplement(out_otsu);
end
end
% imwrite (BW,'Otsus method.jpg')
