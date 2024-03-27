
file                 = xlsread('RVRH.xlsx');
depth                = file(:,1);
RH                   = file(:,2);
RV                   = file(:,3);
CSH_INIT             = file(:,5);
CSH_INIT(CSH_INIT<0) = 0;
CSH_INIT(CSH_INIT>1) = 1;
n                    = size(RH,1);
Rshh                 = 0.60;
Rshv                 = 2.87;
rs                   = file(:,5);
    

%% CG

for tt=1:2155
    a=RV(tt);
    b=RH(tt);
    c=CSH_INIT(tt);
    
    syms Csh_cg Rs_cg f(Csh_cg,Rs_cg) h_Csh h_Rs RHsim_cg RVsim_cg  
    RHsim_cg = 1/(Csh_cg/Rshh + (1-Csh_cg)/Rs_cg);
    RVsim_cg = Csh_cg*Rshv + (1-Csh_cg)*Rs_cg;
    
    f(Csh_cg,Rs_cg)=((a-RVsim_cg))^2+((b-RHsim_cg))^2;
    
    der(Csh_cg,Rs_cg)=gradient(f, [Csh_cg, Rs_cg]);
    g=der(Csh_cg,Rs_cg);
    g_Csh(Csh_cg,Rs_cg)=g(1);
    g_RS(Csh_cg,Rs_cg)=g(2);
    
    der2csh(Csh_cg,Rs_cg)=gradient(g_Csh, [Csh_cg, Rs_cg]);
    h1=der2csh(Csh_cg,Rs_cg);
    h_csh(Csh_cg,Rs_cg)=h1(1);
    h_cshRs(Csh_cg,Rs_cg)=h1(2);
    
    der2rs(Csh_cg, Rs_cg)=gradient(g_RS, [Csh_cg, Rs_cg]);
    h2=der2rs(Csh_cg, Rs_cg);
    h_rs(Csh_cg,Rs_cg)=h2(2);
    
    x0=[c;a];
    f_old=10000;
    niteration=100;
    f_array=[];
    f_new=[];
    xx=zeros(size(x0,1),niteration);
    xx(:,1)=x0;
    r=-[double(g_Csh(x0(1,1),x0(2,1))); double(g_RS(x0(1,1),x0(2,1)))];
    p=r;
    
    for i=1:niteration
        grad_xx=[double(g_Csh(xx(1,i),xx(2,i)));double(g_RS(xx(1,i),xx(2,i)))];
        H=[double(h_csh(xx(1,i),xx(2,i))),double(h_cshRs(xx(1,i),xx(2,i)));double(h_cshRs(xx(1,i),xx(2,i))), double(h_rs(xx(1,i),xx(2,i)))];
        rtr=r'*r;
        alpha=-grad_xx'*p/(p'*H*p);
        xx(:,i+1)=xx(:,i)+alpha*p;
        
        if xx(1,i+1)>1
            xx(1,i+1)=0.9999;
        end
        if xx(1,i+1)<0
            xx(1,i+1)=0.0001;
        end
        
        grad_xx1=[double(g_Csh(xx(1,i+1),xx(2,i+1)));double(g_RS(xx(1,i+1),xx(2,i+1)))];
    %    r=r-alpha*grad_xx'*p;
    %    r=r-alpha*A*p;
        r=-grad_xx1;
    
        beta=r'*r/rtr;
        p=r+beta*p;
        d(:,i)=r;
        f_new(i)=double(f(xx(1,i+1),xx(2,i+1)));
        if (f_new(i)>f_old)
            sprintf('Solution diverges at iteration %.1f ',i)
            f_array(i)=f_old;
            break
        end
        f_old=f_new(i);
        f_array(i)=f_old;
        if i>5
            if and(and(f_array(i-4)-f_array(i-3)<eps,f_array(i-3)-f_array(i-2)<eps),and(f_array(i-2)-f_array(i-1)<eps,f_array(i-1)-f_array(i)<eps))
                sprintf('Solution did not improve after %.1f ',i-4)
                break
            end
        end
            
    end
    sol(:,tt)=xx(:,i-1);
    cost(tt)=f_array(i);
end


%% plot
figure
semilogx(sol(2,1:2155),depth(1:2155))
set(gca,'Ydir','Reverse');
ylim([9600 10200])
grid on

%% END