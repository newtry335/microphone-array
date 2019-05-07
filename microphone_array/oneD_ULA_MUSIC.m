%%һά����
%clear all
close all
clc
%tic %������ʱ��
derad = pi/180;        % deg -> rad�� 1��=����/180�㣩rad
radeg = 180/pi;     %1 rad=180��/��
twpi = 2*pi;
%% ���ֳ�ֵ
kelm = 8;               % ��Ԫ����
dd = 0.5;               % ��Դ���
d=0:dd:(kelm-1)*dd;     % ��0��ʼ��0.5һ����Ԫ��һֱ���ڰ˸���Ԫ
iwave = 3;              % number of DOA  �ź�Դ��
theta = [12 30 60];     % �Ƕ�  ���﷽�� ����Ƕ�
% iwave = 4;              % number of DOA  �ź�Դ��
% theta = [10 30 60 50];     % �Ƕ�  ���﷽�� ����Ƕ�

snr = 10;               % input SNR (dB)�������
n = 500;                 % ������
%% ��������ģ�ͣ����ź�ģ��
A=exp(-1i*twpi*d.'*sin(theta*derad));%%%% direction matrix����ʸ��,��������
S=randn(iwave,n);       %�ź�Դ�źţ�3*500ά��̬�ֲ��������
X=A*S;                  %�����ź�
X1=awgn(X,snr,'measured');   %���ź�X1��Ӹ�˹�������������snr��λdB,���ǰ�����ź�X����dBW
%% ����Э�������
r2=X1';  %�������ת�ã������������Э����
Rxx=X1*X1'/n;   %����Э�������,������Э�������Ĺ���ֵ,�������һЩ
%Rxx=cov(r2);   %���ַ���Ҳ����Э������������������ͬ����ע��Ҫ�ȰѾ���ת��
%% ����ֵ�ֽ⣬���Ұ�����ֵ����
[EV,D]=eig(Rxx);%%%% ����ֵ�ֽ⣬����Rxx������ֵ�Խ���D�������������������ɵľ���EV
EVA=diag(D)';%��ȡ����D�Խ�Ԫ�ع�������EVA������EVA�Ĺ���ת�þ���(�����ǲ��ǹ���û�����𣬶���ʵ��)
% EVA1=diag(D)';
% EVA2=diag(D).';
% EVA3=diag(D);
[EVA,I]=sort(EVA);%����ֵ��С�������򣬽�EVA���򣬲�����һ����EVAͬ�͵ľ���I����IΪ������ָ���¾����еĸ�Ԫ����ԭ�����е�λ�ã�
% [B,I]=sort(EVA1);
% EVAt=fliplr(EVA1);
EVA=fliplr(EVA);%���ҷ�ת���Ӵ�С����
EV=fliplr(EV(:,I));%������ʸ������
%% MUSIC
%tic
angle=zeros(1,361);  %����Ԥ����һ���ڴ棬��������ʱ�䣨��û�У�
SP=zeros(1,361);     %��û�Ż�����
SP2=zeros(1,361);

for iang = 1:361 %ͨ���仯�������ռ���
        angle(iang)=(iang-181)/2;%������һ��1*361�ľ��󣬴�-90�㵽+90��
        phim=derad*angle(iang); %�ӽǶ�ת���ɻ���
        a=exp(-1i*twpi*d*sin(phim)).'; %�źŵ���������
        %a=exp(-j*twpi*d*sin(phim))';%���������Ļ�����������෴
        L=iwave;    %�ź�Դ��
        En=EV(:,L+1:kelm);%����Դ��+1һֱ����Ԫ�������������������������ӿռ䣨ǰ��Դ������������Ϊ�ź��ӿռ䣩
        SP(iang)=(a'*a)/(a'*(En*En')*a);%SP��angleһһ��Ӧ������õ��Ľ����Ϊ����
        %%���еĿռ��׺���
        % 
        % $$P_{MUSIC}(\theta)=\frac{1}{\alpha^H(\theta)*U_N*U_N^H*\alpha(\theta)}$$
        % 
        
end
%toc  
%% �����ļ���ֵ
SP=abs(SP);%����SP��ģ�����ڱȽϴ�С�������ֵ
 %SP1=abs(SP2);
SPmax=max(SP);%��SPΪһ�����򷵻ظ�Ԫ���е����ֵ�����SPΪ�����򷵻ظ���Ԫ�����ֵ���ɵ�������
%SPmax2=max(SP1);
SP=10*log10(SP/SPmax);
[Te,A]=sort(SP);  %�˴�����
[l,p]=findpeaks(SP,iwave);%�����׷庯��
WM=angle(A(359:361));%�˴�Ϊ����õĽǶȣ�ѡ��ǰ3���
WM1=angle(l);
%% ��ͼ
save('data.mat','angle','SP')  %��������
%fig = figure;                 
%fig.Position = get(0,'ScreenSize');%��ͼƬ���

h=plot(angle,SP); %����Ƕȣ��������
hold on
plot(WM1,p,'r*');
set(h,'Linewidth',2) %�����߿�
xlabel('angle (degree)')
ylabel('magnitude (dB)')
axis([-90 90 -60 10])
set(gca, 'XTick',(-90:30:90))
grid on  
zoom on %��ͼ��򿪷Ŵ���


%disp(A)
%toc   %��ʾ����ʱ��
% load handel;
% sound(y,Fs);



