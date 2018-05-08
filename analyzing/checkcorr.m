clear all;
close all;
clc

dir=dir('C:\Users\meisu\Desktop\Windows\NEW-Functions\*.JPG');

I1=imread(dir(1).name);
I2=imread(dir(2).name);

            R=corr2(I1,I2);