function [m,remeshed] = RemeshCtrl(m,Fi,rLim,varargin)

ip = inputParser;
ip.CaseSensitive = false;
ip.addRequired('m', @(x) isa(x,'ModMembrane'));
ip.addRequired('Fi', @(x) isstruct(x));
ip.addRequired('rLim', @(x) isnumeric(x));
ip.addParameter('print_or_not', false, @islogical);
ip.addParameter('l0', [], @isnumeric);
ip.parse(m,Fi,rLim,varargin{:});
%----------------------------------------------------------------------------------------
print_or_not=ip.Results.print_or_not;
%----------------------------------------------------------------------------------------
%%
rDone=false;
if m.pm.remeshScheme==0
    Vpm=m.pm.Vdw;
else
    Vpm=m.pm.Vdh;
end
id_all = (1:m.var.n_edg)';  
remeshed=false;
l=ip.Results.l0;
if isempty(l)
    l = sqrt(sum(([m.var.coord(m.var.edge_all(:,2),1),m.var.coord(m.var.edge_all(:,2),2),m.var.coord(m.var.edge_all(:,2),3)]...
        -[m.var.coord(m.var.edge_all(:,1),1),m.var.coord(m.var.edge_all(:,1),2),m.var.coord(m.var.edge_all(:,1),3)]).^2,2));
end
while rDone==false
    m_save=m;
    idTooLong = l>Vpm.rl_max;
    idTooLong=id_all(idTooLong);
    nTooLong=numel(idTooLong);
    idTooShort = l<Vpm.rl_min;
    idTooShort=id_all(idTooShort);
    nTooShort=numel(idTooShort);
    if (nTooLong==0) && (nTooShort==0)
        rDone=true;
    else
        if print_or_not==true
            fprintf('long: %d short: %d\n',nTooLong,nTooShort);
        end
        rIdxAll=[idTooLong;idTooShort];
        rTypAll=[zeros(nTooLong,1);ones(nTooShort,1)];
        idRand=randsample(nTooLong+nTooShort,1);
        rIdx=rIdxAll(idRand);
        rTyp=rTypAll(idRand);
%==========================================================================
            if rTyp==0
%--------------------------------------------------------------------------
               [m,relaxed] = ModMembrane8LocRelax(m,Fi,m.var.edge_all(rIdx,:),'local',2);
               if relaxed==2
                if print_or_not==true
                    fprintf('failed: initial extension\n');
                end
               elseif relaxed==1
                   [m,remeshed,edg_add] = Remesh(m,rTyp,rIdx,rLim);
                   if remeshed==true
                       [m,~] = ModMembrane8LocRelax(m,Fi,edg_add,'local',1);
                   else
                       m=m_save;
                       [~,id_ring_edg,~,~]=ModMembrane8Ring(m,rIdx,'ring_ord', 1);
                       [m,~] = ModMembrane8LocRelax(m,Fi,m.var.edge_all([rIdx;id_ring_edg],:),'local',1);
                       if print_or_not==true
                        fprintf('failed: unsplittable\n');
                       end
                   end
               end
%--------------------------------------------------------------------------     
            elseif rTyp==1
               [m,relaxed] = ModMembrane8LocRelax(m,Fi,m.var.edge_all(rIdx,:),'local',3);
               if relaxed==2
                if print_or_not==true
                    fprintf('failed: initial merging\n');
                end
               elseif relaxed==1
                   [m,remeshed,edg_add] = Remesh(m,rTyp,rIdx,rLim);
                   if remeshed==true
                       [m,~] = ModMembrane8LocRelax(m,Fi,edg_add,'local',1);
                   else
                       m=m_save;
                       [~,id_ring_edg,~,~]=ModMembrane8Ring(m,rIdx,'ring_ord', 1);
                       [m,~] = ModMembrane8LocRelax(m,Fi,m.var.edge_all([rIdx;id_ring_edg],:),'local',1);
                       if print_or_not==true
                        fprintf('failed: unmergable\n');
                       end
                   end
               end
            end      

        l = sqrt(sum(([m.var.coord(m.var.edge_all(:,2),1),m.var.coord(m.var.edge_all(:,2),2),m.var.coord(m.var.edge_all(:,2),3)]...
        -[m.var.coord(m.var.edge_all(:,1),1),m.var.coord(m.var.edge_all(:,1),2),m.var.coord(m.var.edge_all(:,1),3)]).^2,2));
    end
end
end
%--------------------------------------------------------------------------