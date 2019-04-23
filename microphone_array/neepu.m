clear all
close all
clc

tic

[pyr,fs]=audioread('neepu.wav');%读取音频
%sound(pyr,fs);%播放音频
s=pyr;
fp=bpFilter(s);%带通滤波300-3400Hz
M=5;  %阵元数
N=length(s);%采样点数
c=340; %信号的传播速度
f0=fs;          %采样率，中心频率
fj=1000;        %聚焦频率
lamda=c/f0;
d=0.04;         %阵元间距
snr_dB=-5;      %信噪比
snr=10^(snr_dB/10);%线性信噪比
sir_dB=-5;
sir=10^(sir_dB/10);
theta_s=0*pi/180;%信号到达方向
theta_i1=45*pi/180;
angle=[theta_s theta_i1];
derad=pi/180;

power_s=0;
for t=1:N
    power_s=power_s+(s(t))^2;
end
power_s=power_s/N;         %信号源能量
power_i=power_s/sir;
power_n=power_s/snr;       %噪声信号能量
noise=0.15*wgn(M,N,power_n);%噪声信号
%固定波束形成
tao1=d*sin(theta_s)/c;
tao2=d*sin(theta_i1)/c;
Ts=1.399/N;
L1=ceil(tao1/Ts);
L2=ceil(tao2/Ts);

s1=s';
i1=0.5*s1;

x1=s1+i1+noise(1,:);  %带噪信号
x1bp=bpFilter(x1);
x2=[zeros(1,L1),s1(1:N-L1)]+[zeros(1,L2),i1(1:N-L2)]+noise(2,:);  %各麦克风接收到的信号
x3=[zeros(1,2*L1),s1(1:N-2*L1)]+[zeros(1,2*L2),i1(1:N-2*L2)]+noise(3,:);
x4=[zeros(1,3*L1),s1(1:N-3*L1)]+[zeros(1,3*L2),i1(1:N-3*L2)]+noise(4,:);
x5=[zeros(1,4*L1),s1(1:N-4*L1)]+[zeros(1,4*L2),i1(1:N-4*L2)]+noise(5,:);
X1=[x1;x2;x3;x4;x5];
X2=1/15*(x1+x2+x3+x4+x5);

e=X2-s1;
ps1=sum((s1).^2)/N;
pnout=sum((e).^2)/N;
snr1=10*log10(ps1/pnout)
pnout1=sum(i1).^2/N+power_n;
snr2=10*log10(ps1/pnout1);
snr3=snr1-snr2;
%--------------------------------------------------------------时域图像------------------------------------------------------------------%
figure(1)
subplot(3,1,1);
plot(s1);
title('原始语音');

subplot(3,1,2);
plot(s1+i1+noise(1,:));
title('带噪语音');

% subplot(3,1,3);
% plot(real(X2));
% title('固定波束法增强语音')

subplot(3,1,3);
plot(fp);
title('固定波束法增强语音')
%--------------------------------------------------------------傅里叶变换------------------------------------------------------------------%
pyr_f=fft(pyr,N); %进行傅里叶变换,初始信号
x11=fft(x1,N);  %进行傅里叶变换,带噪信号
X22=fft(X2,N);  %进行傅里叶变换,处理后信号
fp11=fft(fp,N); %进行傅里叶变换,初始信号滤波
x1bp_f=fft(x1bp,N);
%--------------------------------------------------------------频域图像------------------------------------------------------------------%
figure(2)
subplot(3,1,1);
plot(abs(fftshift(pyr_f)));
title('初始信号频谱');
xlabel('Frequency');
ylabel('幅度');
grid on

subplot(3,1,2);
subplot(3,1,2);
plot(abs(fftshift(x11)));
title('带噪信号频谱'); 
xlabel('Time(t)');
ylabel('X(t)');
grid on

subplot(3,1,3);
plot(abs(fftshift(X22)));
title('处理后信号频谱');
xlabel('Frequency');
ylabel('幅度');
grid on

figure(3)
% y=bandp(X22,300,400,200,600,0.1,30,fs);
% plot(abs(fftshift(y)));
% title('处理后信号频谱');
% xlabel('Frequency');
% ylabel('幅度');
% grid on
plot(abs(fftshift(x1bp_f)));
title('滤波信号频谱');
xlabel('Frequency');
ylabel('幅度');
grid on
zoom on
%--------------------------------------------------------------生成语音文件------------------------------------------------------------------%
% audiowrite('test.wav',X2,fs);
% audiowrite('test2.wav',x1,fs);
% audiowrite('fp.wav',fp,fs);
audiowrite('x1bp.wav',x1bp,fs);

toc
