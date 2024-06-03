function [loc_relaxed,m] = locDyn(m,Fi,varargin)

ip = inputParser;
ip.CaseSensitive = false;
ip.addRequired('m', @(x) isa(x,'ModMembrane'));
ip.addRequired('f', @(x) isstruct(x));
ip.addParameter('mex_avail', true, @islogical);
ip.addParameter('local', 0, @isnumeric); %0: global; 1: regular local; 2: extension; 3: shrinkage
ip.addParameter('edg_exo', [], @isnumeric);%for extending or merging the one edge to cross barriers
ip.addParameter('D', 10, @isnumeric);
ip.addParameter('D2', 1000, @isnumeric);
ip.addParameter('sMax', 0.1, @isnumeric);
ip.addParameter('dt', 1e-3, @isnumeric);
ip.addParameter('nt', 100, @isnumeric);
ip.addParameter('plot_or_not', false, @islogical);
ip.parse(m,Fi,varargin{:});
%----------------------------------------------------------------------------------------
edg_exo=ip.Results.edg_exo;
sMax=ip.Results.sMax;
dt=ip.Results.dt;
nt=ip.Results.nt;
Non=numel(m.var.id_on_coord);
mex_avail=ip.Results.mex_avail;
local=ip.Results.local;
D=ip.Results.D;
plot_or_not=ip.Results.plot_or_not;
%----------------------------------------------------------------------------------------
if m.pm.remeshScheme==0
    Vpm=m.pm.Vdw;
else
    Vpm=m.pm.Vdh;
end
%----------------------------------------------------------------------------------------

    %%
    % A:edge
    edgTem=[m.var.id_on_edg;edg_exo];
    A=m.var.edge_all(edgTem,:);
    [C,~,ic]=unique(A);
    C2=(1:numel(C))';
    A=C2(ic);
    A=reshape(A,[numel(A)/2,2]);
    
    idOrg=[m.var.id_on_coord;m.var.id_bound];
    
    [idNew,~]=sort(idOrg);
    
    [~,iNtoO] = ismember(idOrg,idNew);
    id_on_coord=iNtoO(1:Non);
    
    coord=m.var.coord(idNew,:);
    
    i_shift=Fi.rn(1)/m.pm.dr-1;
    
    rmSeed=floor(rand(1,1)*1000+0.5);
    
%     pmc=[sMax,dt,nt,i_shift,m.pm.dr,Non,Vpm.rl_min,Vpm.rl_max,local,D,m.pm.mu,Vpm.rd_min,m.pm.Vdw.rs_max,rmSeed];
    pmc=[sMax,dt,nt,i_shift,m.pm.dr,Non,Vpm.r_best_min,Vpm.r_best_max,local,D,m.pm.mu,Vpm.rd_min,m.pm.Vdh.rs_max,rmSeed];
    %----------------------------------------------------------------------
    %%
    if plot_or_not==true
    d = (coord(A(:,2),:) - coord(A(:,1),:));
    r = sqrt(sum(d.^2,2));
    u = d./r;
    i_shift=Fi.rn(1)/m.pm.dr-1;
    i = floor(r/m.pm.dr+0.5)-i_shift;
    f_edg=Fi.fn(i).*u;
    n_coord=size(coord,1); 
    fc=zeros(n_coord,3);
for i_coord = 1:Non
    icOn=id_on_coord(i_coord);
    fc(icOn,:) = fc(icOn,:)-sum(f_edg(A(:,1)==icOn,:),1);
    fc(icOn,:) = fc(icOn,:)-sum(f_edg(A(:,1)==icOn,:),1);
end
    hold on;
    scatter3(coord(:,1),coord(:,2),coord(:,3),'filled');hold on;
    scatter3(coord(id_on_coord,1),coord(id_on_coord,2),coord(id_on_coord,3),'filled');hold on;
    quiver3(coord(A(:,1),1),coord(A(:,1),2),coord(A(:,1),3),f_edg(:,1),f_edg(:,2),f_edg(:,3)); hold on;
    plot3(coord(A(end,:),1),coord(A(end,:),2),coord(A(end,:),3),'linewidth',2);
    xlimAdj=[xlim,ylim,zlim];
    xlimAdj=[min(xlimAdj),max(xlimAdj)];
    xlim(xlimAdj);ylim(xlimAdj);zlim(xlimAdj);
    hold on;
    for i=1:numel(r)
        plot3([coord(A(i,2),1); coord(A(i,1),1)],[coord(A(i,2),2); coord(A(i,1),2)],[coord(A(i,2),3); coord(A(i,1),3)],'linewidth',2);hold on;
    end
%     scatter3(coord(A(end,1),1),coord(A(end,1),2),coord(A(end,1),3),'filled');hold on;
%     scatter3(coord(A(end,2),1),coord(A(end,2),2),coord(A(end,2),3),'filled');hold on;
%     quiver3(coord(A(end,1),1),coord(A(end,1),2),coord(A(end,1),3),f_edg(end,1),f_edg(end,2),f_edg(end,3));
fprintf('~~~~~ %d\n',local);
pause;
    end
    %%
    %----------------------------------------------------------------------
%     coord=coord_save;
    [coord,loc_relaxed,fc,check]=locDynMex...
       (coord,...
        id_on_coord,...
        A,...
        pmc,...
        Fi.fn,...
        Fi.R,...
        Fi.in,...
        Fi.rg);
    check=floor(check+0.5);
    if ((check==1) && (m.pm.remeshScheme==0))
        disp('out of boarder!');
        disp('out of boarder!');
        pause;
    end
    %%
%     scatter3(coord(:,1),coord(:,2),coord(:,3),'filled');hold on;
%     quiver3(coord(:,1),coord(:,2),coord(:,3),fc(:,1),fc(:,2),fc(:,3));
    loc_relaxed=floor(loc_relaxed+0.5);
    m.var.coord(idNew,:)=coord;
end


