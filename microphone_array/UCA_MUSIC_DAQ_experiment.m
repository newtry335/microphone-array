%%
clc
clear
close all

fs  = 50000;
N   =  5000;
M   =   8;%阵元数

%% 读取数据
tic
x1  = xlsread('D:\synfloder\OneDrive\OneDrive - kolo\gra_proj\microphone_array\8.xls','B2:B5001');
x2  = xlsread('D:\synfloder\OneDrive\OneDrive - kolo\gra_proj\microphone_array\8.xls','D2:D5001');
x3  = xlsread('D:\synfloder\OneDrive\OneDrive - kolo\gra_proj\microphone_array\8.xls','F2:F5001');
x4  = xlsread('D:\synfloder\OneDrive\OneDrive - kolo\gra_proj\microphone_array\8.xls','H2:H5001');
x5  = xlsread('D:\synfloder\OneDrive\OneDrive - kolo\gra_proj\microphone_array\8.xls','J2:J5001');
x6  = xlsread('D:\synfloder\OneDrive\OneDrive - kolo\gra_proj\microphone_array\8.xls','L2:L5001');
x7  = xlsread('D:\synfloder\OneDrive\OneDrive - kolo\gra_proj\microphone_array\8.xls','N2:N5001');
x8  = xlsread('D:\synfloder\OneDrive\OneDrive - kolo\gra_proj\microphone_array\8.xls','P2:P5001');
toc
%9秒多
%% MUSIC_UCA_验证
imag=sqrt(-1);
dw=0.88;  % 半径波长比
sn=1;     % 信号个数
X=[x1';x2';x3';x4';x5';x6';x7';x8'];
R=X*X'/N;                   % 8*8求协方差矩阵
[tzxiangliang,tzzhi]=eig(R);% [V,D]=eig(A)：求矩阵A的全部特征值，构成对角阵D，并求A的特征向量构成V的列向量
%tzxiangliang 特征向量8*8=64个
%tzzhi  特征值构成对角矩阵  8个
Nspace=tzxiangliang(:,1:M-sn);%噪声子空间对应小的特征值（从小到大排列）???a(:,:,1)是一个三维矩阵，a(:,:,1)表示取a矩阵第一页的所有行和列
%Nspace  8*8=64个
AQ1=zeros(8,1);
P=zeros(90,180);
for azi=1:1:360
    for ele=1:1:90
        for m=1:M
            AQ1(m,1)=exp(1i*2*pi*dw*cos(azi*pi/180-2*pi*(m-1)/M)*sin(ele*pi/180));%8行1
            %disp(AQ1);
        end
        % disp(AQ1);
        Power=1/(AQ1'*(Nspace*Nspace')*AQ1);
        % disp(Power);
        P(ele,azi)=abs(Power);   %90*180
    end
    
end
toc
% [PH,PHI]=max(P(:));
% PL=min(P(:));
% RP=20*log10(PH/PL);
% i=1:1:257;%这里别使用for循环，慢
% p(i)=sqrt(PL^2+(i-1)*(PH^2-PL^2)/256);
% if p>p(257)
% i(p)=256;
% elseif p<p(1)
%         i(p)=1;
% else
%     i(p)=k;
% end
% figure;
% % Pmax=max(max(P)) ;  %找出P的最大值zmax
% [Pmax,xi_max]=max(P);
% [Pmax,yi_max]=max(Pmax);
%% 绘图
%第一幅图
[PHI_col,PHI_row]=find(P==max(P(:)));%寻找最大值对应的行列
figure(1)
contourf(P);%等高线图
hold on
plot(PHI_row,PHI_col,'k+')%把最大值点的位置画出来
colorbar %添加色标
title('UCA MUSIC：M=8 ；d/lamda=0.85 ');
xlabel('azi');ylabel('ele');
grid on
%第二幅图
figure(2)
h=pcolor(P);%(伪色彩图)热度图
axis off
set(h,'edgecolor','none','facecolor','interp');%去掉网格，平滑热度图
hold on
%plot(PHI_row,PHI_col,'k+')%把最大值点的位置画出来
colormap jet%颜色图风格
%% 图像处理
%无框保存
set(gca,'position',[0 0 1 1])
saveas(h,'C:\Users\MannixWong\Desktop\UCA_MUSIC.jpg');
%去蓝底
I=imread('C:\Users\MannixWong\Desktop\UCA_MUSIC.jpg');%读取
G=rgb2gray(I);%转为灰度图
ima=imadjust(G,[0.1,0.21],[]);
bw=imbinarize(ima);
% figure(3)
% imshow(bw)
level=graythresh(G);
bw2=imbinarize(ima,level);%转化为二值化图像
% figure,imshow(bw2)
bw3=~bw2;%二值化图像取反
bw4=bwareaopen(bw3,40);
% figure,imshow(bw4)
bw5=~bw4;
% figure,imshow(bw5)
R=I(:,:,1);
G=I(:,:,2);
B=I(:,:,3);
%背景设为白色
R(~bw5)=255;   
G(~bw5)=255;      
B(~bw5)=255;
rgb=cat(3,R,G,B);
% figure,imshow(rgb)
siz=size(I);
alpha=ones(siz(1),siz(2));
alpha(B==255)=0;%设置蓝色部分为透明
imwrite(rgb,'test.png','Alpha',alpha)
[E,map,alpha]=imread('D:\synfloder\OneDrive\OneDrive - kolo\gra_proj\microphone_array\test.png');%读取
%第三张图
figure(3)
Z=imshow(E);
%调整大小
J=imresize(E,[600,960]);%图像处理最终效果
%% 图像融合
obj=videoinput('winvideo',1);  %一般的家用摄像头第三个参数这样就可以，不能运行直接去掉第三个参数也可以             
h1=preview(obj);              %显示摄像头
h2=figure(4);                    %新建显示图像figure,同时获取句柄
while ishandle(h1) && ishandle(h2)          %两个句柄有一个关闭就结束程序
    frame=getsnapshot(obj);     %捕获图像
    K=imresize(frame,[600,960]);
    M = imadd(0.3*J,K);
     imshow(M);              %显示图像
    drawnow;           
end



