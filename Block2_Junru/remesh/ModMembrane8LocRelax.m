function [m,loc_relaxed] = ModMembrane8LocRelax(m,Fi,edg_add, varargin)

ip = inputParser;
ip.CaseSensitive = false;
ip.addRequired('m', @(x) isa(x,'ModMembrane'));
ip.addRequired('Fi', @(x) isstruct(x));
ip.addRequired('edg_add', @(x) isnumeric(x));
ip.addParameter('plot_or_not', false, @islogical);
ip.addParameter('dt', 0.001, @isnumeric);
ip.addParameter('n_step', 100000, @isnumeric);
ip.addParameter('D', 100, @isnumeric);
ip.addParameter('f_const_only', true, @islogical);
ip.addParameter('local', 0, @isnumeric); %0: global; 1: regular local; 2: extension; 3: shrinkage
ip.addParameter('i_typ', 2, @isnumeric);
ip.addParameter('i_int', 1, @isnumeric);
ip.addParameter('ring_ord', 1, @isnumeric);
ip.parse(m,Fi,edg_add, varargin{:});
%--------------------------------------------------------------------------------------------------------
plot_or_not = ip.Results.plot_or_not;
f_const_only=ip.Results.f_const_only;
local=ip.Results.local;
i_typ=ip.Results.i_typ;
i_int=ip.Results.i_int;
ring_ord=ip.Results.ring_ord;
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%%
edg_add_org=edg_add;
n_add = size(edg_add,1);
id_ring_ver = [];
id_ring_edg = [];
id_all = 1:m.var.n_edg;
id_edg_add = zeros(n_add,1);
for i = 1:n_add
    i_edg = sum(m.var.edge_all==edg_add(i,1),2)+sum(m.var.edge_all==edg_add(i,2),2);
    i_edg = id_all(i_edg==2);
    id_edg_add(i) = i_edg;
    if ring_ord==1
        [id_ring_ver_tem,id_ring_edg_tem] = ModMembrane8Ring(m,i_edg,'ring_ord',ring_ord);
    elseif ring_ord==2
        [~,~,id_ring_ver_tem,id_ring_edg_tem] = ModMembrane8Ring(m,i_edg,'ring_ord',ring_ord);
    else
        error('ring_ord is wrong');
    end
    id_ring_ver = [id_ring_ver; id_ring_ver_tem];
    id_ring_edg = [id_ring_edg; id_ring_edg_tem];
    %%---------------------------------------------------------------------
%     coord=m.var.coord;
%     edg=m.var.edge_all;
%     scatter3(coord(id_ring_ver,1),coord(id_ring_ver,2),coord(id_ring_ver,3),'filled');hold on;
%     scatter3(coord(edg_add,1),coord(edg_add,2),coord(edg_add,3),'filled');hold on;
%     for iii=1:numel(id_ring_edg)
%     plot3(coord(edg(id_ring_edg(iii),:),1),coord(edg(id_ring_edg(iii),:),2),coord(edg(id_ring_edg(iii),:),3));hold on;
%     end
    %%---------------------------------------------------------------------
end
for i = 1:n_add
    id_ring_edg(id_ring_edg==id_edg_add(i)) = [];
    id_ring_ver(id_ring_ver==edg_add(i,1) | id_ring_ver==edg_add(i,2)) = [];
end
id_ring_edg = unique(id_ring_edg);
id_ring_edg = [id_ring_edg; id_edg_add];
id_ring_ver = unique(id_ring_ver);
n_ring_edg = size(id_ring_edg,1);
n_ring_ver = size(id_ring_ver,1);
id_all = 1:m.var.n_coord;
id_ring_in=unique(id_all(m.var.edge_all(id_ring_edg,:)));
for i=1:n_ring_ver
    id_ring_in(id_ring_in==id_ring_ver(i))=[];
end

if local>1
    edg_add=[edg_add;m.var.edge_all(id_ring_edg,:)];
    edg_add = unique(edg_add,'row');
    n_add = size(edg_add,1);
id_ring_ver = [];
id_ring_edg = [];
id_all = 1:m.var.n_edg;
id_edg_add = zeros(n_add,1);
for i = 1:n_add
    i_edg = sum(m.var.edge_all==edg_add(i,1),2)+sum(m.var.edge_all==edg_add(i,2),2);
    i_edg = id_all(i_edg==2);
    id_edg_add(i) = i_edg;
    if ring_ord==1
        [id_ring_ver_tem,id_ring_edg_tem] = ModMembrane8Ring(m,i_edg,'ring_ord',ring_ord);
    elseif ring_ord==2
        [~,~,id_ring_ver_tem,id_ring_edg_tem] = ModMembrane8Ring(m,i_edg,'ring_ord',ring_ord);
    else
        error('ring_ord is wrong');
    end
    id_ring_ver = [id_ring_ver; id_ring_ver_tem];
    id_ring_edg = [id_ring_edg; id_ring_edg_tem];
end
for i = 1:n_add
    id_ring_edg(id_ring_edg==id_edg_add(i)) = [];
    id_ring_ver(id_ring_ver==edg_add(i,1) | id_ring_ver==edg_add(i,2)) = [];
end
id_ring_edg = unique(id_ring_edg);
id_ring_edg = [id_ring_edg; id_edg_add];
id_ring_ver = unique(id_ring_ver);
n_ring_edg = size(id_ring_edg,1);
n_ring_ver = size(id_ring_ver,1);
id_all = 1:m.var.n_coord;
id_ring_in=unique(id_all(m.var.edge_all(id_ring_edg,:)));
for i=1:n_ring_ver
    id_ring_in(id_ring_in==id_ring_ver(i))=[];
end
end

id_on_edg_save=m.var.id_on_edg;
n_on_edg_save=m.var.n_on_edg;
id_on_coord_save=m.var.id_on_coord;
n_on_coord_save=m.var.n_on_coord;
id_bound_save=m.var.id_bound;
n_bound_save=m.var.n_bound;
id_all = 1:m.var.n_edg;
if local >1
%    edg_add_org_flip=[edg_add_org(2) edg_add_org(1)];
    id_tem=sum(abs(m.var.edge_all-edg_add_org),2)==0;
    edg_exo=id_all(id_tem);
else
    edg_exo=[];
end
m.var.id_on_edg=id_ring_edg;
m.var.id_on_coord=id_ring_in;
m.var.id_bound=id_ring_ver;
[id_int,i_int]=intersect(m.var.id_on_coord,id_bound_save);
if ~isempty(id_int)
    m.var.id_on_coord(i_int)=[];
    m.var.id_bound=[m.var.id_bound;id_int];
end
%%
[loc_relaxed,m] = locDyn(m,Fi,'edg_exo',edg_exo,'nt',ip.Results.n_step,'local',local);

m.var.id_on_edg=id_on_edg_save;
m.var.n_on_edg=n_on_edg_save;
m.var.id_on_coord=id_on_coord_save;
m.var.n_on_coord=n_on_coord_save;
m.var.id_bound=id_bound_save;
m.var.n_bound=n_bound_save;
%%
% nt = 2000;
% s_max = 0.01;
% kBT = 10000;
% kBT2 = 1000;
% f = zeros(m.var.n_ver,3);
% r_std_all = [pm.f.r_std_n(pm.f.r_sub_n(1:end-1)), pm.f.r_std_n(pm.f.r_sub_n(2:end))];
% R = random('Stable',1,1,1,0,[10000,1]);
% id_tem = (1:max(size(pm.f.r_sub_n))-1)';
% id_all = zeros(size(id_tem,1),n_ring_edg);
% for i=1:n_ring_edg
%     id_all(:,i) = id_tem;
% end
% relaxed = false;
% for it = 1:nt
%     %fprintf('%d,',it);
%     f = f*0;
%     r = sqrt(sum((m.var.ver(m.var.edge_all(id_ring_edg,2),:) - m.var.ver(m.var.edge_all(id_ring_edg,1),:)).^2,2));
%     u = (m.var.ver(m.var.edge_all(id_ring_edg,2),:)-m.var.ver(m.var.edge_all(id_ring_edg,1),:))./r;
%     dr_tem1 = r'-r_std_all(:,1);
%     dr_tem2 = r'-r_std_all(:,2);
%     iseg = id_all(dr_tem1.*dr_tem2<0);
%     ir = pm.f.r_sub_n(iseg);
%     f(m.var.edge_all(id_ring_edg,1),:) = f(m.var.edge_all(id_ring_edg,1),:)-pm.f.f_n(ir).*u;
%     f(m.var.edge_all(id_ring_edg,2),:) = f(m.var.edge_all(id_ring_edg,2),:)+pm.f.f_n(ir).*u;
%     %----------------------------------------------------------------------
%     id_tem1 = r<pm.Vdw.rl_min;  id_tem2 = r>pm.Vdw.rl_max;
%     n_tem1 = length(id_tem1(id_tem1)); n_tem2 = length(id_tem2(id_tem2));
%     if n_tem1 > 0
%         f_rand = randsample(R,n_tem1).*u(id_tem1,:)*kBT+randn(n_tem1,3)*kBT2;
%         f(m.var.edge_all(id_ring_edg(id_tem1),1),:) = f(m.var.edge_all(id_ring_edg(id_tem1),1),:)-f_rand;
%         f(m.var.edge_all(id_ring_edg(id_tem1),2),:) = f(m.var.edge_all(id_ring_edg(id_tem1),2),:)+f_rand;
%     end
%     if n_tem2 > 0
%         f_rand = -randsample(R,n_tem2).*u(id_tem2,:)*kBT+randn(n_tem2,3)*kBT2;
%         f(m.var.edge_all(id_ring_edg(id_tem2),1),:) = f(m.var.edge_all(id_ring_edg(id_tem2),1),:)-f_rand;
%         f(m.var.edge_all(id_ring_edg(id_tem2),2),:) = f(m.var.edge_all(id_ring_edg(id_tem2),2),:)+f_rand;
%     end
%     if (n_tem1 == 0) && (n_tem2 == 0)
%         relaxed = true;
%         break;
%     end
%     %fprintf('%d, %d, %d\n',it, n_tem1,n_tem2);
%     %----------------------------------------------------------------------
%     f(id_ring_ver,:) = f(id_ring_ver,:)*0;
%     f_mag = max([sqrt(sum(f(m.var.edge_all(id_ring_edg,1),:).^2,2));sqrt(sum(f(m.var.edge_all(id_ring_edg,2),:).^2,2))]);
%     if f_mag*pm.dt > s_max
%         dt = s_max/f_mag;
%     else
%         dt = pm.dt;
%     end
%     m.var.ver = m.var.ver+f*dt;
% end
%fprintf('\n');

%%
if (plot_or_not == true) %|| (relaxed == false)
    %save('temp.mat','var','edg_add');

% m.var.id_on_edg=id_ring_edg;
% m.var.id_on_coord=id_ring_in;
% m.var.id_bound=id_ring_ver;

plot(m,'FaceAlpha', 0.5, 'LineStyle','none'); xlabel('x');ylabel('y');zlabel('z');hold on;
scatter3(m.var.coord(edg_add(:,1),1),m.var.coord(edg_add(:,1),2),m.var.coord(edg_add(:,1),3),40,'filled','MarkerFaceColor',[0 1 0]); hold on;
scatter3(m.var.coord(edg_add(:,2),1),m.var.coord(edg_add(:,2),2),m.var.coord(edg_add(:,2),3),40,'filled','MarkerFaceColor',[0 1 0]); hold on;
scatter3(m.var.coord(m.var.id_bound,1),m.var.coord(m.var.id_bound,2),m.var.coord(m.var.id_bound,3),40,'filled','MarkerFaceColor',[0 1 1]); hold on;
scatter3(m.var.coord(m.var.id_on_coord,1),m.var.coord(m.var.id_on_coord,2),m.var.coord(m.var.id_on_coord,3),40,'filled','MarkerFaceColor',[0 0.2 1]); hold on;

if local>1
    plot3([m.var.coord(m.var.edge_all(edg_exo,1),1),m.var.coord(m.var.edge_all(edg_exo,2),1)],...
      [m.var.coord(m.var.edge_all(edg_exo,1),2),m.var.coord(m.var.edge_all(edg_exo,2),2)],...
      [m.var.coord(m.var.edge_all(edg_exo,1),3),m.var.coord(m.var.edge_all(edg_exo,2),3)],'linewidth',5,'color',[1 0 0]);hold on;
end

for i = 1:n_add
    plot3([m.var.coord(edg_add(i,1),1),m.var.coord(edg_add(i,2),1)],...
      [m.var.coord(edg_add(i,1),2),m.var.coord(edg_add(i,2),2)],...
      [m.var.coord(edg_add(i,1),3),m.var.coord(edg_add(i,2),3)],'linewidth',3,'color',[0 1 0]);hold on;
end

for i = 1:n_ring_edg
    plot3([m.var.coord(m.var.edge_all(m.var.id_on_edg(i),1),1),m.var.coord(m.var.edge_all(m.var.id_on_edg(i),2),1)],...
      [m.var.coord(m.var.edge_all(m.var.id_on_edg(i),1),2),m.var.coord(m.var.edge_all(m.var.id_on_edg(i),2),2)],...
      [m.var.coord(m.var.edge_all(m.var.id_on_edg(i),1),3),m.var.coord(m.var.edge_all(m.var.id_on_edg(i),2),3)],'linewidth',2,'color',[1 1 0]);hold on;
end
end
