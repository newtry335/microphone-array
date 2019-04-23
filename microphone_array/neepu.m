clear all
close all
clc

tic

[pyr,fs]=audioread('neepu.wav');%��ȡ��Ƶ
%sound(pyr,fs);%������Ƶ
s=pyr;
fp=bpFilter(s);%��ͨ�˲�300-3400Hz
M=5;  %��Ԫ��
N=length(s);%��������
c=340; %�źŵĴ����ٶ�
f0=fs;          %�����ʣ�����Ƶ��
fj=1000;        %�۽�Ƶ��
lamda=c/f0;
d=0.04;         %��Ԫ���
snr_dB=-5;      %�����
snr=10^(snr_dB/10);%���������
sir_dB=-5;
sir=10^(sir_dB/10);
theta_s=0*pi/180;%�źŵ��﷽��
theta_i1=45*pi/180;
angle=[theta_s theta_i1];
derad=pi/180;

power_s=0;
for t=1:N
    power_s=power_s+(s(t))^2;
end
power_s=power_s/N;         %�ź�Դ����
power_i=power_s/sir;
power_n=power_s/snr;       %�����ź�����
noise=0.15*wgn(M,N,power_n);%�����ź�
%�̶������γ�
tao1=d*sin(theta_s)/c;
tao2=d*sin(theta_i1)/c;
Ts=1.399/N;
L1=ceil(tao1/Ts);
L2=ceil(tao2/Ts);

s1=s';
i1=0.5*s1;

x1=s1+i1+noise(1,:);  %�����ź�
x1bp=bpFilter(x1);
x2=[zeros(1,L1),s1(1:N-L1)]+[zeros(1,L2),i1(1:N-L2)]+noise(2,:);  %����˷���յ����ź�
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
%--------------------------------------------------------------ʱ��ͼ��------------------------------------------------------------------%
figure(1)
subplot(3,1,1);
plot(s1);
title('ԭʼ����');

subplot(3,1,2);
plot(s1+i1+noise(1,:));
title('��������');

% subplot(3,1,3);
% plot(real(X2));
% title('�̶���������ǿ����')

subplot(3,1,3);
plot(fp);
title('�̶���������ǿ����')
%--------------------------------------------------------------����Ҷ�任------------------------------------------------------------------%
pyr_f=fft(pyr,N); %���и���Ҷ�任,��ʼ�ź�
x11=fft(x1,N);  %���и���Ҷ�任,�����ź�
X22=fft(X2,N);  %���и���Ҷ�任,������ź�
fp11=fft(fp,N); %���и���Ҷ�任,��ʼ�ź��˲�
x1bp_f=fft(x1bp,N);
%--------------------------------------------------------------Ƶ��ͼ��------------------------------------------------------------------%
figure(2)
subplot(3,1,1);
plot(abs(fftshift(pyr_f)));
title('��ʼ�ź�Ƶ��');
xlabel('Frequency');
ylabel('����');
grid on

subplot(3,1,2);
subplot(3,1,2);
plot(abs(fftshift(x11)));
title('�����ź�Ƶ��'); 
xlabel('Time(t)');
ylabel('X(t)');
grid on

subplot(3,1,3);
plot(abs(fftshift(X22)));
title('������ź�Ƶ��');
xlabel('Frequency');
ylabel('����');
grid on

figure(3)
% y=bandp(X22,300,400,200,600,0.1,30,fs);
% plot(abs(fftshift(y)));
% title('������ź�Ƶ��');
% xlabel('Frequency');
% ylabel('����');
% grid on
plot(abs(fftshift(x1bp_f)));
title('�˲��ź�Ƶ��');
xlabel('Frequency');
ylabel('����');
grid on
zoom on
%--------------------------------------------------------------���������ļ�------------------------------------------------------------------%
% audiowrite('test.wav',X2,fs);
% audiowrite('test2.wav',x1,fs);
% audiowrite('fp.wav',fp,fs);
audiowrite('x1bp.wav',x1bp,fs);

toc
