function [fi] = Finternal(m, varargin)

ip = inputParser;
ip.CaseSensitive = false;
ip.addRequired('m', @(x) isa(x,'ModMembrane'));
ip.addParameter('plot_or_not', false, @islogical);
ip.parse(m, varargin{:});
%--------------------------------------------------------------------------------------------------------
dr = m.pm.dr;
plot_or_not = ip.Results.plot_or_not;
rsq_std = m.pm.f_const_rsq_std; 
std_std = m.pm.f_const_std_std;%*m.pm.k_const;
%%
fi =  struct(  ...
               'fn', [],'Vn', [], 'rn', [], 'nr', [], 'in', [],'rg', [],'ig', [],'ng', [],'R', []...
);%fi-internal force; fn-fine scaled force; rn-fine scale; nr-numel of rn; rg-grouped scale; ng-numel of rg
%%
fi.R = random('Stable',1,1,1,0,[10000,1]);
%--------------------------------------------------------------------------------------------------------
%%
if m.pm.remeshScheme==0
    Vpm=m.pm.Vdw;
else
    Vpm=m.pm.Vdh;
end
rmax = Vpm.r_2;
fi.rn = (Vpm.r_1+dr:dr:Vpm.r_2-dr);fi.rn = fi.rn';
fi.nr = max(size(fi.rn,1));
Vn = zeros(fi.nr,1);
for i = 1:fi.nr
Vn(i) = m.Vinternal(fi.rn(i),Vpm,m.pm.remeshScheme);
end
%%
i_sad = 1; %saddle point
eps = 0.0;
for i = 2:fi.nr-1
    if ((Vn(i-1) < Vn(i)-eps) && (Vn(i+1) < Vn(i)-eps)) || ((Vn(i-1) > Vn(i)+eps) && (Vn(i+1) > Vn(i)+eps))
        i_sad = [i_sad;i];
    end
end
i_sad = [i_sad;fi.nr];
if plot_or_not
    figure;
plot(fi.rn,Vn); hold on;
plot(fi.rn(i_sad),Vn(i_sad),'*'); hold on;
%ylim([0 8])
end
%%
fi.ig=i_sad;
seg=[fi.ig(1:end-1),fi.ig(2:end)];
n_seg=size(seg,1);
ig_new=fi.ig;
settle=false(n_seg,1);
while numel(settle(settle==false))>0
    min_rsq=inf; std_max=0;
    settle=false(n_seg,1);
for i_seg=1:n_seg
    if seg(i_seg,2)-seg(i_seg,1)>1
    y = Vn(seg(i_seg,1):seg(i_seg,2));
    x = fi.rn(seg(i_seg,1):seg(i_seg,2));
    p = polyfit(x,y,1);
    yfit =  p(1) * x + p(2);
    yresid  = y - yfit;
    SSresid = sum(yresid.^2);
    SStotal = (length(y)-1) * var(y);
    rsq = 1 - SSresid/SStotal;
    std_y=std(y);
    if rsq<min_rsq
        min_rsq=rsq;
    end 
    if std_y>std_max
        std_max=std_y;
    end
    if ((rsq < rsq_std) || (std_y > std_std))
        ig_new=[ig_new;floor(0.5*(seg(i_seg,1)+seg(i_seg,2))+0.5)];
        ig_new=sort(ig_new);
    else
        settle(i_seg) = true;
    end
    else
        settle(i_seg) = true;
    end
end
fi.ig=ig_new;
seg=[fi.ig(1:end-1),fi.ig(2:end)];
n_seg=size(seg,1);
end
fprintf('membrane constant force settled: %d, %f, %f\n',n_seg, min_rsq, std_max);
%%
fi.ng=size(fi.ig,1);
k=zeros(fi.ng-1,1);
d=zeros(fi.ng-1,1);
Vtem=zeros(fi.ng-1,1);
fi.Vn = zeros(fi.nr,1);
for i = 1:fi.ng-1
y = [Vn(fi.ig(i)),Vn(fi.ig(i+1))];
x = [fi.rn(fi.ig(i)),fi.rn(fi.ig(i+1))];
k(i) = (y(2)-y(1))/(x(2)-x(1)); 
d(i)=(y(2)+y(1) -k(i)*(x(2)+x(1)))*0.5;
Vtem(i)=Vn(fi.ig(i+1));
fi.Vn(fi.ig(i):fi.ig(i+1)-1) = d(i)+k(i)*fi.rn(fi.ig(i):fi.ig(i+1)-1);
end
fi.Vn(end)=Vn(end);

fi.fn = zeros(fi.nr,1);
fi.in = zeros(fi.nr,1);
for i = 2:fi.ng
    fi.fn(fi.ig(i-1)+1:fi.ig(i)) = -k(i-1);
    fi.in(fi.ig(i-1)+1:fi.ig(i)) = i;
end
fi.fn(1)=fi.fn(2);
fi.in(1)=fi.in(2);

fi.rg=fi.rn(fi.ig);

if plot_or_not

% plot(fi.rn,Vn); hold on;
% % plot(fi.rn(i_sad),Vn(i_sad),'*');hold on;
% plot(fi.rn(fi.ig),Vn(fi.ig),'o');
for i = 1:fi.ng-1
    plot(fi.rn(fi.ig(i):fi.ig(i+1)),fi.Vn(fi.ig(i):fi.ig(i+1)),'-','linewidth',2); hold on;
end
% subplot(1,2,2);
%plot(fi.rn,f_std); hold on;
%ylim([-70 70]);
end
%==============================================================================
%==============================================================================
end
%==============================================================================

