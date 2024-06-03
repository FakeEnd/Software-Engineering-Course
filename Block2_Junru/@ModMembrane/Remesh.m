function [m,remeshed,edg_add] = Remesh(m,rmTyp,rmIdx,rLim,varargin)

ip = inputParser;
ip.CaseSensitive = false;
ip.addRequired('m', @(x) isa(x,'ModMembrane'));
ip.addRequired('rmTyp', @(x) isnumeric(x)); %0: split; 1:merge
ip.addRequired('rmIdx', @(x) isnumeric(x));
ip.addRequired('rLim', @(x) isnumeric(x));
ip.addParameter('plot_or_not', false, @islogical);
ip.parse(m,rmTyp,rmIdx,rLim,varargin{:});
%----------------------------------------------------------------------------------------
plot_or_not=ip.Results.plot_or_not;
%----------------------------------------------------------------------------------------
%%
if plot_or_not==true
    fig=figure('units','normalized','outerposition',[0 0 1 1]); subplot(1,2,1); title('before');
    plot(m,'f',fig); hold on;
    plot3([m.var.coord(m.var.edge_all(rmIdx,1),1);m.var.coord(m.var.edge_all(rmIdx,2),1)],...
          [m.var.coord(m.var.edge_all(rmIdx,1),2);m.var.coord(m.var.edge_all(rmIdx,2),2)],...
          [m.var.coord(m.var.edge_all(rmIdx,1),3);m.var.coord(m.var.edge_all(rmIdx,2),3)],'-','color',[0 0 1],'linewidth',3);
end
if numel(rLim)~=2
    error('rLim dimension wrong!');
elseif rLim(1) > rLim(2)
    error('rLim direction wrong! r1 < r2 required');
end
%%
var=struct('n_edg',m.var.n_edg,...
           'face_unq',m.var.face_unq,...
           'edge_all',m.var.edge_all,...
           'val',m.var.val,...
           'n_ver',m.var.n_coord);
i_edg=rmIdx;      
[~,id_ring_edg,~,~]=Ring(m,i_edg,'ring_ord', 1);
if numel(intersect(id_ring_edg,m.var.id_on_edg))==numel(id_ring_edg) 
id_tem = sum(var.face_unq == var.edge_all(i_edg,1),2)+sum(var.face_unq == var.edge_all(i_edg,2),2);
id_tem = var.face_unq(id_tem==2,:);
id_tem2 = id_tem(1,:);
[~,i_tem1] = min(abs(id_tem2-var.edge_all(i_edg,1)));
[~,i_tem2] = min(abs(id_tem2-var.edge_all(i_edg,2)));
i_tem1 = i_tem1 + 1;
if i_tem1 > 3
    i_tem1 = 1;
end
j = zeros(2,1);
if i_tem1 == i_tem2
    j(1) = id_tem2(id_tem2 ~= var.edge_all(i_edg,1) & id_tem2 ~= var.edge_all(i_edg,2));
    id_tem2 = id_tem(2,:);
    j(2) = id_tem2(id_tem2 ~= var.edge_all(i_edg,1) & id_tem2 ~= var.edge_all(i_edg,2));
else
    id_tem2 = id_tem(2,:);
    j(1) = id_tem2(id_tem2 ~= var.edge_all(i_edg,1) & id_tem2 ~= var.edge_all(i_edg,2));
    id_tem2 = id_tem(1,:);
    j(2) = id_tem2(id_tem2 ~= var.edge_all(i_edg,1) & id_tem2 ~= var.edge_all(i_edg,2));
end
%--------------------------------------------------------------------------
k = zeros(4,1);
id_tem = sum(var.face_unq == var.edge_all(i_edg,1),2)+sum(var.face_unq == j(1),2);
id_tem = var.face_unq(id_tem==2,:);
k(1) = id_tem(id_tem~=var.edge_all(i_edg,1) & id_tem~=var.edge_all(i_edg,2) & id_tem ~= j(1));
id_tem = sum(var.face_unq == var.edge_all(i_edg,2),2)+sum(var.face_unq == j(1),2);
id_tem = var.face_unq(id_tem==2,:);
k(2) = id_tem(id_tem~=var.edge_all(i_edg,1) & id_tem~=var.edge_all(i_edg,2) & id_tem ~= j(1));
id_tem = sum(var.face_unq == var.edge_all(i_edg,2),2)+sum(var.face_unq == j(2),2);
id_tem = var.face_unq(id_tem==2,:);
k(3) = id_tem(id_tem~=var.edge_all(i_edg,1) & id_tem~=var.edge_all(i_edg,2) & id_tem ~= j(2));
id_tem = sum(var.face_unq == var.edge_all(i_edg,1),2)+sum(var.face_unq == j(2),2);
id_tem = var.face_unq(id_tem==2,:);
k(4) = id_tem(id_tem~=var.edge_all(i_edg,1) & id_tem~=var.edge_all(i_edg,2) & id_tem ~= j(2));
%----------------------------------------------------------------------------------------       
if rmTyp==0
    split = struct('can',[],'edg_add',[],'j',[],'k',[],'id_ring_edg',[]);
    split.can = true;
    split.id_ring_edg = id_ring_edg;
    split.j = j;
    split.k = k;
    i_n = var.n_ver+1;
%--------------------------------------------------------------------------    
edg_add = ones(16,8); val_tem = [var.val(var.edge_all(i_edg,1:2))'-2,4,var.val(j)'-1,var.val(k)'];
val_new = val_tem.*ones(16,1);
%--------------------------------------------------------------------------
edg_add(1:8,1) = k(1); edg_add(1:8,2) = i_n;
edg_add(9:16,1) = var.edge_all(i_edg,1); edg_add(9:16,2) = j(1);
val_new(1:8,6) = val_new(1:8,6)+1; val_new(1:8,3) = val_new(1:8,3)+1;
val_new(9:16,1) = val_new(9:16,1)+1; val_new(9:16,4) = val_new(9:16,4)+1;
%--------------------------------------------------------------------------
edg_add([1:4,9:12]',3) = k(2); edg_add([1:4,9:12]',4) = i_n;
edg_add([5:8,13:16]',3) = var.edge_all(i_edg,2); edg_add([5:8,13:16]',4) = j(1);
val_new([1:4,9:12]',7) = val_new([1:4,9:12]',7)+1; val_new([1:4,9:12]',3) = val_new([1:4,9:12]',3)+1;
val_new([5:8,13:16]',2) = val_new([5:8,13:16]',2)+1; val_new([5:8,13:16]',4) = val_new([5:8,13:16]',4)+1;
%--------------------------------------------------------------------------
edg_add([1:2,5:6,9:10,13:14]',5) = k(3); edg_add([1:2,5:6,9:10,13:14]',6) = i_n;
edg_add([3:4,7:8,11:12,15:16]',5) = var.edge_all(i_edg,2); edg_add([3:4,7:8,11:12,15:16]',6) = j(2);
val_new([1:2,5:6,9:10,13:14]',8) = val_new([1:2,5:6,9:10,13:14]',8)+1; val_new([1:2,5:6,9:10,13:14]',3) = val_new([1:2,5:6,9:10,13:14]',3)+1;
val_new([3:4,7:8,11:12,15:16]',2) = val_new([3:4,7:8,11:12,15:16]',2)+1; val_new([3:4,7:8,11:12,15:16]',5) = val_new([3:4,7:8,11:12,15:16]',5)+1;
%--------------------------------------------------------------------------
edg_add(1:2:16,7) = k(4); edg_add(1:2:16,8) = i_n;
edg_add(2:2:16,7) = var.edge_all(i_edg,1); edg_add(2:2:16,8) = j(2);
val_new(1:2:16,9) = val_new(1:2:16,9) +1; val_new(1:2:16,3) = val_new(1:2:16,3) +1;
val_new(2:2:16,1) = val_new(2:2:16,1)+1; val_new(2:2:16,5) = val_new(2:2:16,5)+1;
%--------------------------------------------------------------------------
    can_split = true(16,1);
    can_split(sum((val_new>m.pm.n_val_max) | (val_new<m.pm.n_val_min),2)>0) = false;
    if isempty(can_split(can_split==true))
        split.can = false;
    else
        split.edg_add = edg_add(can_split==true,:);
    end
%----------------------------------------------------------------------------------------    
elseif rmTyp==1
    merge = struct('can',[],'edg_add',[],'j',[],'k',[],'id_ring_edg',[]);
    merge.can = true;
    merge.id_ring_edg = id_ring_edg;
    merge.j = j;
    merge.k = k;
%--------------------------------------------------------------------------
edg_add = ones(4,4); 
val_tem = [sum(var.val(var.edge_all(i_edg,1:2)))-8+2,var.val(j)'-2,var.val(k)'];
val_new = val_tem.*ones(4,1);
%--------------------------------------------------------------------------
edg_add(1:2,1)=k(1); edg_add(1:2,2)=k(2);
edg_add(3:4,1)=var.edge_all(i_edg,1); edg_add(3:4,2) = j(1);
val_new(1:2,4) = val_new(1:2,4)+1; val_new(1:2,5) = val_new(1:2,5)+1;
val_new(3:4,1) = val_new(3:4,1)+1; val_new(3:4,2) = val_new(3:4,2)+1;
%--------------------------------------------------------------------------
edg_add(1,3)=k(3); edg_add(1,4)=k(4);
val_new(1,6) = val_new(1,6)+1; val_new(1,7) = val_new(1,7)+1;
edg_add(2,3)=var.edge_all(i_edg,1); edg_add(2,4)=j(2);
val_new(2,1) = val_new(2,1)+1; val_new(2,3) = val_new(2,3)+1;
%--------------------------------------------------------------------------
edg_add(3,3)=k(3); edg_add(3,4)=k(4);
val_new(3,6) = val_new(3,6)+1; val_new(3,7) = val_new(3,7)+1;
edg_add(4,3)=var.edge_all(i_edg,1); edg_add(4,4)=j(2);
val_new(4,1) = val_new(4,1)+1; val_new(4,3) = val_new(4,3)+1;
%--------------------------------------------------------------------------
can_merge = true(4,1);
can_merge(sum((val_new>m.pm.n_val_max) | (val_new<m.pm.n_val_min),2)>0) = false;
if isempty(can_merge(can_merge==true))
    merge.can = false;
else
    merge.edg_add = edg_add(can_merge==true,:);
end
%--------------------------------------------------------------------------
%----------------------------------------------------------------------------------------    
end
%----------------------------------------------------------------------------------------
else
    split.can = false;
    merge.can = false;
end
%%
remeshed=true;
if (rmTyp==0) && (split.can==false)
    remeshed=false;
elseif (rmTyp==1) && (merge.can==false)
    remeshed=false;
end
if remeshed==true
    if rmTyp==0
                var_new=m.var;
                var_new.coord = [m.var.coord;0.5*(m.var.coord(m.var.edge_all(i_edg,1),:)+m.var.coord(m.var.edge_all(i_edg,2),:))];
                var_new.n_coord = var_new.n_coord+1;
%----------------------------------------------------------------------
                j = split.j; k = split.k; id_ring_edg = split.id_ring_edg; i_n = m.var.n_coord+1; i_e = m.var.edge_all(i_edg,:);
%----------------------------------------------------------------------
                edg_rem = [i_edg;id_ring_edg(sum(m.var.edge_all(id_ring_edg,1)==j(1),2)+sum(m.var.edge_all(id_ring_edg,2)==j(1),2) > 0);...
                    id_ring_edg(sum(m.var.edge_all(id_ring_edg,1)==j(2),2)+sum(m.var.edge_all(id_ring_edg,2)==j(2),2) > 0)];
                id_rem = false(m.var.n_edg,1);
                id_rem(edg_rem) = true;
                var_new.edge_all(id_rem,:) = [];
                edg_tem = split.edg_add;
                r_tem = [sqrt(sum((var_new.coord(edg_tem(:,1),:)-var_new.coord(edg_tem(:,2),:)).^2,2)),...
                    sqrt(sum((var_new.coord(edg_tem(:,3),:)-var_new.coord(edg_tem(:,4),:)).^2,2)),...
                    sqrt(sum((var_new.coord(edg_tem(:,5),:)-var_new.coord(edg_tem(:,6),:)).^2,2)),...
                    sqrt(sum((var_new.coord(edg_tem(:,7),:)-var_new.coord(edg_tem(:,8),:)).^2,2))];
%                 [~,id_tem] = min(std(r_tem-m.pm.l0,[],2));
                [~,id_sort] = sort(std(r_tem-m.pm.l0,[],2));
                n_try=numel(id_sort);
                successTem=false;
                edg_save=edg_tem;
                var_new_save=var_new;
                mSave=m;
                for i_try=1:n_try
                id_tem=id_sort(i_try);
                edg_tem = edg_save(id_tem,:);
                edg_tem = [edg_tem(1:2);edg_tem(3:4);edg_tem(5:6);edg_tem(7:8)];
                edg_add = [edg_tem;[m.var.edge_all(i_edg,1) i_n];[i_n m.var.edge_all(i_edg,2)];[i_n j(1)];[i_n j(2)]];
                var_new.edge_all = [var_new.edge_all;edg_add];
%----------------------------------------------------------------------
                face_rem = zeros(6,1);
                n_face = size(m.var.face_unq,1);
                id_all = 1:n_face;
                i_tem = sum(m.var.face_unq==i_e(1),2) + sum(m.var.face_unq==j(1),2) + sum(m.var.face_unq==k(1),2); face_rem(1) = id_all(i_tem==3);
                i_tem = sum(m.var.face_unq==i_e(1),2) + sum(m.var.face_unq==i_e(2),2) + sum(m.var.face_unq==j(1),2); face_rem(2) = id_all(i_tem==3);
                i_tem = sum(m.var.face_unq==i_e(2),2) + sum(m.var.face_unq==k(2),2) + sum(m.var.face_unq==j(1),2); face_rem(3) = id_all(i_tem==3);
                i_tem = sum(m.var.face_unq==i_e(2),2) + sum(m.var.face_unq==j(2),2) + sum(m.var.face_unq==k(3),2); face_rem(4) = id_all(i_tem==3);
                i_tem = sum(m.var.face_unq==i_e(1),2) + sum(m.var.face_unq==i_e(2),2) + sum(m.var.face_unq==j(2),2); face_rem(5) = id_all(i_tem==3);
                i_tem = sum(m.var.face_unq==i_e(1),2) + sum(m.var.face_unq==j(2),2) + sum(m.var.face_unq==k(4),2); face_rem(6) = id_all(i_tem==3);
                id_rem = false(n_face,1);
                id_rem(face_rem) = true;
                var_new.face_unq(id_rem,:) = [];
%--------------------------------------------------------------------------
                face_add = zeros(8,3);
                i_f = 1;
                if sum(edg_add(i_f,:) == i_e(1),2)+sum(edg_add(i_f,:) == i_e(2),2) == 1
                    face_add(i_f,:) = [i_e(1) i_n j(1)]; face_add(i_f+1,:) = [j(1) k(1) i_e(1)];
                else
                    face_add(i_f,:) = [i_e(1) i_n k(1)]; face_add(i_f+1,:) = [i_n j(1) k(1)];
                end
                i_f = 2;
                if sum(edg_add(i_f,:) == i_e(1),2)+sum(edg_add(i_f,:) == i_e(2),2) == 1
                    face_add(i_f*2-1,:) = [i_n i_e(2) j(1)]; face_add(i_f*2,:) = [i_e(2) k(2) j(1)];
                else
                    face_add(i_f*2-1,:) = [i_n i_e(2) k(2)]; face_add(i_f*2,:) = [k(2) j(1) i_n];
                end
                i_f = 3;
                if sum(edg_add(i_f,:) == i_e(1),2)+sum(edg_add(i_f,:) == i_e(2),2) == 1
                    face_add(i_f*2-1,:) = [i_e(2) i_n j(2)]; face_add(i_f*2,:) = [i_e(2) j(2) k(3)];
                else
                    face_add(i_f*2-1,:) = [i_e(2) i_n k(3)]; face_add(i_f*2,:) = [k(3) i_n j(2)];
                end
                i_f = 4;
                if sum(edg_add(i_f,:) == i_e(1),2)+sum(edg_add(i_f,:) == i_e(2),2) == 1
                    face_add(i_f*2-1,:) = [i_n i_e(1) j(2)]; face_add(i_f*2,:) = [j(2) i_e(1) k(4)];
                else
                    face_add(i_f*2-1,:) = [i_n i_e(1) k(4)]; face_add(i_f*2,:) = [i_n k(4) j(2)];
                end
                var_new.face_unq = [var_new.face_unq;face_add];
%--------------------------------------------------------------------------
                [m.var] = AddVertex(m,m.pm,m.var,var_new.coord,var_new.edge_all,var_new.face_unq);
                r = sqrt(sum(([m.var.coord(m.var.edge_all(:,2),1),m.var.coord(m.var.edge_all(:,2),2),m.var.coord(m.var.edge_all(:,2),3)]...
                  -[m.var.coord(m.var.edge_all(:,1),1),m.var.coord(m.var.edge_all(:,1),2),m.var.coord(m.var.edge_all(:,1),3)]).^2,2));
                if (max(r)<rLim(2)) && (min(r)>rLim(1))
                    successTem=true;
                    break;
                elseif (i_try==n_try)
                    successTem=false;
                end
                m=mSave;
                var_new=var_new_save;
                end
                if successTem==true
                    [m] = getUface(m);
                else
                    remeshed=false;
                    m=mSave;
                end
                %%
                if (plot_or_not==true) && (remeshed==true)
                    figure(fig); subplot(1,2,2); title('after splitting');
                    plot(m,'f',fig); hold on; 
                    for i=1:size(edg_add,1)
                    plot3([m.var.coord(edg_add(i,2),1);m.var.coord(edg_add(i,1),1)],...
                          [m.var.coord(edg_add(i,2),2);m.var.coord(edg_add(i,1),2)],...
                          [m.var.coord(edg_add(i,2),3);m.var.coord(edg_add(i,1),3)],'-','color',[0 1 0],'linewidth',3); hold on;
                    end
                end
%--------------------------------------------------------------------------
    elseif rmTyp==1
                var_new=m.var;
                i = m.var.edge_all(i_edg,:);
                var_new.coord(i(1),:) = 0.5*(m.var.coord(i(1),:)+m.var.coord(i(2),:));
%----------------------------------------------------------------------
                j = merge.j; k = merge.k; id_ring_edg = merge.id_ring_edg; i_e = m.var.edge_all(i_edg,:);
%----------------------------------------------------------------------
                edg_rem = [i_edg;...
                           id_ring_edg(sum(m.var.edge_all(id_ring_edg,1)==j(1),2)+sum(m.var.edge_all(id_ring_edg,2)==j(1),2) > 0);...
                           id_ring_edg(sum(m.var.edge_all(id_ring_edg,1)==j(2),2)+sum(m.var.edge_all(id_ring_edg,2)==j(2),2) > 0)];
                id_rem = false(m.var.n_edg,1);
                id_rem(edg_rem) = true;
                var_new.edge_all(id_rem,:) = [];
                edg_tem = merge.edg_add;
                r_tem = [sqrt(sum((var_new.coord(edg_tem(:,1),:)-var_new.coord(edg_tem(:,2),:)).^2,2)),...
                    sqrt(sum((var_new.coord(edg_tem(:,3),:)-var_new.coord(edg_tem(:,4),:)).^2,2))];
%                 [~,id_tem] = min(std(r_tem-m.pm.l0,[],2));
                [~,id_sort] = sort(std(r_tem-m.pm.l0,[],2));
                n_try=numel(id_sort);
                successTem=false;
                edg_save=edg_tem;
                var_new_save=var_new;
                mSave=m;
                for i_try=1:n_try
                id_tem=id_sort(i_try);
                edg_tem = edg_save(id_tem,:);
                edg_tem = [edg_tem(1:2);edg_tem(3:4)];
                edg_add = edg_tem;
                var_new.edge_all = [var_new.edge_all;edg_add];
%----------------------------------------------------------------------
                face_rem = zeros(6,1);
                n_face = size(m.var.face_unq,1);
                id_all = 1:n_face;
                i_tem = sum(m.var.face_unq==i_e(1),2) + sum(m.var.face_unq==j(1),2) + sum(m.var.face_unq==k(1),2); face_rem(1) = id_all(i_tem==3);
                i_tem = sum(m.var.face_unq==i_e(1),2) + sum(m.var.face_unq==i_e(2),2) + sum(m.var.face_unq==j(1),2); face_rem(2) = id_all(i_tem==3);
                i_tem = sum(m.var.face_unq==i_e(2),2) + sum(m.var.face_unq==k(2),2) + sum(m.var.face_unq==j(1),2); face_rem(3) = id_all(i_tem==3);
                i_tem = sum(m.var.face_unq==i_e(2),2) + sum(m.var.face_unq==j(2),2) + sum(m.var.face_unq==k(3),2); face_rem(4) = id_all(i_tem==3);
                i_tem = sum(m.var.face_unq==i_e(1),2) + sum(m.var.face_unq==i_e(2),2) + sum(m.var.face_unq==j(2),2); face_rem(5) = id_all(i_tem==3);
                i_tem = sum(m.var.face_unq==i_e(1),2) + sum(m.var.face_unq==j(2),2) + sum(m.var.face_unq==k(4),2); face_rem(6) = id_all(i_tem==3);
                id_rem = false(n_face,1);
                id_rem(face_rem) = true;
                var_new.face_unq(id_rem,:) = [];
%--------------------------------------------------------------------------
                face_add = zeros(4,3);
                i_f = 1;
                if sum(edg_add(i_f,:) == i_e(1),2) == 1
                    face_add(i_f,:) = [j(1) i_e(1) k(2)]; face_add(i_f+1,:) = [j(1) k(1) i_e(1)];
                else
                    face_add(i_f,:) = [j(1) k(1) k(2)]; face_add(i_f+1,:) = [k(1) i_e(1) k(2)];
                end
                i_f = 2;
                if sum(edg_add(i_f,:) == i_e(1),2) == 1
                    face_add(i_f*2-1,:) = [j(2) k(3) i_e(1)]; face_add(i_f*2,:) = [i_e(1) k(4) j(2)];
                else
                    face_add(i_f*2-1,:) = [j(2) k(3) k(4)]; face_add(i_f*2,:) = [k(4) k(3) i_e(1)];
                end
                
                var_new.face_unq = [var_new.face_unq;face_add];
                
                var_new.face_unq(var_new.face_unq==i_e(2)) = i_e(1);
                id_tem = var_new.face_unq>i_e(2);
                var_new.face_unq(id_tem) = var_new.face_unq(id_tem)-1;
                
                var_new.edge_all(var_new.edge_all==i_e(2)) = i_e(1);
                id_tem = var_new.edge_all>i_e(2);
                var_new.edge_all(id_tem) = var_new.edge_all(id_tem)-1;
                
                var_new.coord(i(2),:) = [];
                var_new.n_coord = var_new.n_coord-1;
                [m.var] = AddVertex(m,m.pm,m.var,var_new.coord,var_new.edge_all,var_new.face_unq,'id_merge',i_e);
                r = sqrt(sum(([m.var.coord(m.var.edge_all(:,2),1),m.var.coord(m.var.edge_all(:,2),2),m.var.coord(m.var.edge_all(:,2),3)]...
                  -[m.var.coord(m.var.edge_all(:,1),1),m.var.coord(m.var.edge_all(:,1),2),m.var.coord(m.var.edge_all(:,1),3)]).^2,2));
                if (max(r)<rLim(2)) && (min(r)>rLim(1))
                    successTem=true;
                    break;
                elseif (i_try==n_try)
                    successTem=false;
                end
                m=mSave;
                var_new=var_new_save;
                end
                if successTem==true
                    [m] = getUface(m);
                    edg_add = m.var.edge_all(end-1:end,:);
                    i_e_new=edg_add(1,1);
                    if (edg_add(2,1)~=i_e_new) && (edg_add(2,2)~=i_e_new)
                        i_e_new=edg_add(1,2);
                    end
                    edg_add=[edg_add;m.var.edge_all(sum(m.var.edge_all == i_e_new,2)>0,:)];
                    edg_add=unique(edg_add,'row');
                else
                    remeshed=false;
                    m=mSave;
                end
                %%
                if (plot_or_not==true) && (remeshed==true)
                    figure(fig); subplot(1,2,2); title('after merging');
                    plot(m,'f',fig); hold on;
                    for i=1:size(edg_add,1)
                    plot3([m.var.coord(edg_add(i,2),1);m.var.coord(edg_add(i,1),1)],...
                          [m.var.coord(edg_add(i,2),2);m.var.coord(edg_add(i,1),2)],...
                          [m.var.coord(edg_add(i,2),3);m.var.coord(edg_add(i,1),3)],'-','color',[0 1 0],'linewidth',3); hold on;
                    end
                end
    end
end
%--------------------------------------------------------------------------