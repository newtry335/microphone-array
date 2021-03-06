%%一维线阵
%clear all
close all
clc
%tic %启动计时器
derad = pi/180;        % deg -> rad， 1°=（Π/180°）rad
radeg = 180/pi;     %1 rad=180°/Π
twpi = 2*pi;
%% 各种初值
kelm = 8;               % 阵元数量
dd = 0.5;               % 阵源间距
d=0:dd:(kelm-1)*dd;     % 从0开始隔0.5一个阵元，一直到第八个阵元
iwave = 3;              % number of DOA  信号源数
theta = [12 30 60];     % 角度  波达方向 入射角度
% iwave = 4;              % number of DOA  信号源数
% theta = [10 30 60 50];     % 角度  波达方向 入射角度

snr = 10;               % input SNR (dB)，信噪比
n = 500;                 % 采样点
%% 构建接收模型，和信号模型
A=exp(-1i*twpi*d.'*sin(theta*derad));%%%% direction matrix方向矢量,阵列流形
S=randn(iwave,n);       %信号源信号，3*500维正态分布随机矩阵
X=A*S;                  %接收信号
X1=awgn(X,snr,'measured');   %向信号X1添加高斯白噪声，信噪比snr单位dB,添加前计算信号X功率dBW
%% 计算协方差矩阵
r2=X1';  %这里进行转置，方便接下来算协方差
Rxx=X1*X1'/n;   %计算协方差矩阵,这里是协方差矩阵的估计值,正常算快一些
%Rxx=cov(r2);   %这种方法也可算协方差矩阵，与上述结果相同，但注意要先把矩阵转置
%% 特征值分解，并且把特征值排序
[EV,D]=eig(Rxx);%%%% 特征值分解，计算Rxx的特征值对角阵D！！！和特征向量构成的矩阵EV
EVA=diag(D)';%抽取矩阵D对角元素构成向量EVA，再求EVA的共轭转置矩阵(这里是不是共轭没有区别，都是实数)
% EVA1=diag(D)';
% EVA2=diag(D).';
% EVA3=diag(D);
[EVA,I]=sort(EVA);%特征值从小到大排序，将EVA排序，并返回一个与EVA同型的矩阵I，（I为索引，指定新矩阵中的各元素在原矩阵中的位置）
% [B,I]=sort(EVA1);
% EVAt=fliplr(EVA1);
EVA=fliplr(EVA);%左右翻转，从大到小排序
EV=fliplr(EV(:,I));%对特征矢量排序
%% MUSIC
%tic
angle=zeros(1,361);  %这里预分配一下内存，可以缩短时间（并没有）
SP=zeros(1,361);     %并没优化多少
SP2=zeros(1,361);

for iang = 1:361 %通过变化θ搜索空间谱
        angle(iang)=(iang-181)/2;%构成了一个1*361的矩阵，从-90°到+90°
        phim=derad*angle(iang); %从角度转化成弧度
        a=exp(-1i*twpi*d*sin(phim)).'; %信号的特征向量
        %a=exp(-j*twpi*d*sin(phim))';%如果不共轭的话，结果正好相反
        L=iwave;    %信号源数
        En=EV(:,L+1:kelm);%从信源数+1一直到阵元数个特征向量，构成了噪声子空间（前信源数个特征向量为信号子空间）
        SP(iang)=(a'*a)/(a'*(En*En')*a);%SP与angle一一对应，这里得到的结果均为复数
        %%阵列的空间谱函数
        % 
        % $$P_{MUSIC}(\theta)=\frac{1}{\alpha^H(\theta)*U_N*U_N^H*\alpha(\theta)}$$
        % 
        
end
%toc  
%% 求函数的极大值
SP=abs(SP);%求复数SP的模，用于比较大小，求最大值
 %SP1=abs(SP2);
SPmax=max(SP);%如SP为一向量则返回各元素中的最大值，如果SP为矩阵则返回各列元素最大值构成的行向量
%SPmax2=max(SP1);
SP=10*log10(SP/SPmax);
[Te,A]=sort(SP);  %此处排序
[l,p]=findpeaks(SP,iwave);%搜索谱峰函数
WM=angle(A(359:361));%此处为计算得的角度，选出前3大的
WM1=angle(l);
%% 绘图
save('data.mat','angle','SP')  %保存数据
%fig = figure;                 
%fig.Position = get(0,'ScreenSize');%把图片最大化

h=plot(angle,SP); %横轴角度，纵轴幅度
hold on
plot(WM1,p,'r*');
set(h,'Linewidth',2) %设置线宽
xlabel('angle (degree)')
ylabel('magnitude (dB)')
axis([-90 90 -60 10])
set(gca, 'XTick',(-90:30:90))
grid on  
zoom on %作图后打开放大功能


%disp(A)
%toc   %显示所用时间
% load handel;
% sound(y,Fs);



