%%
clc
clear
close all

fs  = 50000;
N   =  5000;
M   =   8;%��Ԫ��

%% ��ȡ����
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
%9���
%% MUSIC_UCA_��֤
imag=sqrt(-1);
dw=0.88;  % �뾶������
sn=1;     % �źŸ���
X=[x1';x2';x3';x4';x5';x6';x7';x8'];
R=X*X'/N;                   % 8*8��Э�������
[tzxiangliang,tzzhi]=eig(R);% [V,D]=eig(A)�������A��ȫ������ֵ�����ɶԽ���D������A��������������V��������
%tzxiangliang ��������8*8=64��
%tzzhi  ����ֵ���ɶԽǾ���  8��
Nspace=tzxiangliang(:,1:M-sn);%�����ӿռ��ӦС������ֵ����С�������У�???a(:,:,1)��һ����ά����a(:,:,1)��ʾȡa�����һҳ�������к���
%Nspace  8*8=64��
AQ1=zeros(8,1);
P=zeros(90,180);
for azi=1:1:360
    for ele=1:1:90
        for m=1:M
            AQ1(m,1)=exp(1i*2*pi*dw*cos(azi*pi/180-2*pi*(m-1)/M)*sin(ele*pi/180));%8��1
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
% i=1:1:257;%�����ʹ��forѭ������
% p(i)=sqrt(PL^2+(i-1)*(PH^2-PL^2)/256);
% if p>p(257)
% i(p)=256;
% elseif p<p(1)
%         i(p)=1;
% else
%     i(p)=k;
% end
% figure;
% % Pmax=max(max(P)) ;  %�ҳ�P�����ֵzmax
% [Pmax,xi_max]=max(P);
% [Pmax,yi_max]=max(Pmax);
%% ��ͼ
%��һ��ͼ
[PHI_col,PHI_row]=find(P==max(P(:)));%Ѱ�����ֵ��Ӧ������
figure(1)
contourf(P);%�ȸ���ͼ
hold on
plot(PHI_row,PHI_col,'k+')%�����ֵ���λ�û�����
colorbar %���ɫ��
title('UCA MUSIC��M=8 ��d/lamda=0.85 ');
xlabel('azi');ylabel('ele');
grid on
%�ڶ���ͼ
figure(2)
h=pcolor(P);%(αɫ��ͼ)�ȶ�ͼ
axis off
set(h,'edgecolor','none','facecolor','interp');%ȥ������ƽ���ȶ�ͼ
hold on
%plot(PHI_row,PHI_col,'k+')%�����ֵ���λ�û�����
colormap jet%��ɫͼ���
%% ͼ����
%�޿򱣴�
set(gca,'position',[0 0 1 1])
saveas(h,'C:\Users\MannixWong\Desktop\UCA_MUSIC.jpg');
%ȥ����
I=imread('C:\Users\MannixWong\Desktop\UCA_MUSIC.jpg');%��ȡ
G=rgb2gray(I);%תΪ�Ҷ�ͼ
ima=imadjust(G,[0.1,0.21],[]);
bw=imbinarize(ima);
% figure(3)
% imshow(bw)
level=graythresh(G);
bw2=imbinarize(ima,level);%ת��Ϊ��ֵ��ͼ��
% figure,imshow(bw2)
bw3=~bw2;%��ֵ��ͼ��ȡ��
bw4=bwareaopen(bw3,40);
% figure,imshow(bw4)
bw5=~bw4;
% figure,imshow(bw5)
R=I(:,:,1);
G=I(:,:,2);
B=I(:,:,3);
%������Ϊ��ɫ
R(~bw5)=255;   
G(~bw5)=255;      
B(~bw5)=255;
rgb=cat(3,R,G,B);
% figure,imshow(rgb)
siz=size(I);
alpha=ones(siz(1),siz(2));
alpha(B==255)=0;%������ɫ����Ϊ͸��
imwrite(rgb,'test.png','Alpha',alpha)
[E,map,alpha]=imread('D:\synfloder\OneDrive\OneDrive - kolo\gra_proj\microphone_array\test.png');%��ȡ
%������ͼ
figure(3)
Z=imshow(E);
%������С
J=imresize(E,[600,960]);%ͼ��������Ч��
%% ͼ���ں�
obj=videoinput('winvideo',1);  %һ��ļ�������ͷ���������������Ϳ��ԣ���������ֱ��ȥ������������Ҳ����             
h1=preview(obj);              %��ʾ����ͷ
h2=figure(4);                    %�½���ʾͼ��figure,ͬʱ��ȡ���
while ishandle(h1) && ishandle(h2)          %���������һ���رվͽ�������
    frame=getsnapshot(obj);     %����ͼ��
    K=imresize(frame,[600,960]);
    M = imadd(0.3*J,K);
     imshow(M);              %��ʾͼ��
    drawnow;           
end



