function [id_ring_ver1,id_ring_edg1,id_ring_ver2,id_ring_edg2] = Ring(m,edge_ctr, varargin)

ip = inputParser;
ip.CaseSensitive = false;
ip.addRequired('m', @(x) isa(x,'ModMembrane'));
ip.addRequired('edge_ctr', @(x) isnumeric(x));
ip.addParameter('plot_or_not', false, @islogical);
ip.addParameter('ring_ord', 1, @isnumeric); %1-ring or 2-ring
ip.addParameter('mex_avail', false, @islogical);
ip.parse(m, edge_ctr, varargin{:});
%--------------------------------------------------------------------------------------------------------
plot_or_not = ip.Results.plot_or_not;
%%
if ip.Results.mex_avail==true
    %% 
    [id_ring_ver1,id_ring_edg1,n_ring_ver1,n_ring_edg1]=ch.ModMembrane8RingMex(m.var.edge_all,edge_ctr);
    id_ring_ver1=id_ring_ver1(1:n_ring_ver1);
    id_ring_edg1=id_ring_edg1(1:n_ring_edg1);
    id_ring_ver2=[];id_ring_edg2=[];
    %%
    if ip.Results.ring_ord==2
        %%
        id_ring_ver=id_ring_ver1;
        id_ring_edg=id_ring_edg1;
        n_ring=numel(id_ring_edg);
    for i_ring=1:n_ring
        edge_ctr_add=id_ring_edg1(i_ring);
        [id_ring_ver_add,id_ring_edg_add,n_ring_ver1,n_ring_edg1]=ch.ModMembrane8RingMex(m.var.edge_all,edge_ctr_add);
        id_ring_ver_add=id_ring_ver_add(1:n_ring_ver1);
        id_ring_edg_add=id_ring_edg_add(1:n_ring_edg1);
        
        id_ring_ver=[id_ring_ver;id_ring_ver_add];
        id_ring_edg=[id_ring_edg;id_ring_edg_add]; 
    end
    id_ring_edg(id_ring_edg==edge_ctr)=[];
    id_ring_ver(id_ring_ver==m.var.edge_all(edge_ctr,1) | id_ring_ver==m.var.edge_all(edge_ctr,2)) = [];
    id_ring_ver2=unique(id_ring_ver);id_ring_edg2=unique(id_ring_edg);    
    end
else
%%
id_ring1 = sum(m.var.edge_all == m.var.edge_all(edge_ctr,1),2);
id_ring1 = m.var.edge_all(id_ring1==1,:);
id_ring2 = sum(m.var.edge_all == m.var.edge_all(edge_ctr,2),2);
id_ring2 = m.var.edge_all(id_ring2==1,:);

id_ring_ver = unique([id_ring1; id_ring2]);
id_ring_ver(id_ring_ver==m.var.edge_all(edge_ctr,1) | id_ring_ver==m.var.edge_all(edge_ctr,2)) = [];

id_all = 1:m.var.n_edg;
id_ring1 = sum(m.var.edge_all == m.var.edge_all(edge_ctr,1),2);
id_ring1 = id_all(id_ring1==1);
id_ring2 = sum(m.var.edge_all == m.var.edge_all(edge_ctr,2),2);
id_ring2 = id_all(id_ring2==1);

id_ring_edg = unique([id_ring1'; id_ring2']);
id_ring_edg(id_ring_edg==edge_ctr) = [];

id_ring_ver1=id_ring_ver;id_ring_edg1=id_ring_edg;
id_ring_ver2=[];id_ring_edg2=[];
%%
if ip.Results.ring_ord==2
    %%
    n_ring=numel(id_ring_edg);
    for i_ring=1:n_ring
        edge_ctr_add=id_ring_edg1(i_ring);
        id_ring1 = sum(m.var.edge_all == m.var.edge_all(edge_ctr_add,1),2);
        id_ring1 = m.var.edge_all(id_ring1==1,:);
        id_ring2 = sum(m.var.edge_all == m.var.edge_all(edge_ctr_add,2),2);
        id_ring2 = m.var.edge_all(id_ring2==1,:);
        
        id_ring_ver_add = unique([id_ring1; id_ring2]);
        id_ring_ver_add(id_ring_ver_add==m.var.edge_all(edge_ctr_add,1) | id_ring_ver_add==m.var.edge_all(edge_ctr_add,2)) = [];
        id_ring_ver=[id_ring_ver;id_ring_ver_add];
        
        id_all = 1:m.var.n_edg;
        id_ring1 = sum(m.var.edge_all == m.var.edge_all(edge_ctr_add,1),2);
        id_ring1 = id_all(id_ring1==1);
        id_ring2 = sum(m.var.edge_all == m.var.edge_all(edge_ctr_add,2),2);
        id_ring2 = id_all(id_ring2==1);
        
        id_ring_edg_add = unique([id_ring1'; id_ring2']);
        id_ring_edg_add(id_ring_edg_add==edge_ctr_add) = [];
        id_ring_edg=[id_ring_edg;id_ring_edg_add];
    end
    id_ring_edg(id_ring_edg==edge_ctr)=[];
    id_ring_ver(id_ring_ver==m.var.edge_all(edge_ctr,1) | id_ring_ver==m.var.edge_all(edge_ctr,2)) = [];
    id_ring_ver2=unique(id_ring_ver);id_ring_edg2=unique(id_ring_edg);
end
end
%%
if plot_or_not
plot(m,'FaceAlpha', 0.5, 'LineStyle','none'); xlabel('x');ylabel('y');zlabel('z');hold on;

scatter3(m.var.coord(m.var.edge_all(edge_ctr,:),1),m.var.coord(m.var.edge_all(edge_ctr,:),2),m.var.coord(m.var.edge_all(edge_ctr,:),3),40,'filled'); hold on;

plot3([m.var.coord(m.var.edge_all(edge_ctr,1),1),m.var.coord(m.var.edge_all(edge_ctr,2),1)],...
      [m.var.coord(m.var.edge_all(edge_ctr,1),2),m.var.coord(m.var.edge_all(edge_ctr,2),2)],...
      [m.var.coord(m.var.edge_all(edge_ctr,1),3),m.var.coord(m.var.edge_all(edge_ctr,2),3)],'linewidth',2,'color',[0 1 1]);hold on;
for i = 1:size(id_ring_edg1,1)
    plot3([m.var.coord(m.var.edge_all(id_ring_edg1(i),1),1),m.var.coord(m.var.edge_all(id_ring_edg1(i),2),1)],...
          [m.var.coord(m.var.edge_all(id_ring_edg1(i),1),2),m.var.coord(m.var.edge_all(id_ring_edg1(i),2),2)],...
          [m.var.coord(m.var.edge_all(id_ring_edg1(i),1),3),m.var.coord(m.var.edge_all(id_ring_edg1(i),2),3)],'linewidth',2,'color',[0 1 0.5]);hold on;
end
hold on;

for i = 1:size(id_ring_edg2,1)
    plot3([m.var.coord(m.var.edge_all(id_ring_edg2(i),1),1),m.var.coord(m.var.edge_all(id_ring_edg2(i),2),1)],...
          [m.var.coord(m.var.edge_all(id_ring_edg2(i),1),2),m.var.coord(m.var.edge_all(id_ring_edg2(i),2),2)],...
          [m.var.coord(m.var.edge_all(id_ring_edg2(i),1),3),m.var.coord(m.var.edge_all(id_ring_edg2(i),2),3)],'linewidth',2,'color',[0.5 1 0]);hold on;
end
scatter3(m.var.coord(m.var.edge_all(edge_ctr,:),1),m.var.coord(m.var.edge_all(edge_ctr,:),2),m.var.coord(m.var.edge_all(edge_ctr,:),3),40,'filled'); hold on;

scatter3(m.var.coord(id_ring_ver2,1),m.var.coord(id_ring_ver2,2),m.var.coord(id_ring_ver2,3),40,'filled','markerfacecolor',[0 0 1]); hold on;
scatter3(m.var.coord(id_ring_ver1,1),m.var.coord(id_ring_ver1,2),m.var.coord(id_ring_ver1,3),40,'filled','markerfacecolor',[1 0 0]); hold on;
end
