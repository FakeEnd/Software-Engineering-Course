
function [obj] = SetVar(obj, varargin)

ip = inputParser;
ip.CaseSensitive = false;
ip.addRequired('obj', @(x) isobject(x));
ip.addParameter('update', false, @islogical);
ip.addParameter('coord', [], @isnumeric);
ip.addParameter('face', [], @isnumeric);
ip.addParameter('var_old', [], @isstruct);
ip.addParameter('init_perturb', true, @islogical);
ip.addParameter('k_perturb', 0.0001, @isnumeric);
ip.parse(obj, varargin{:});
%--------------------------------------------------------------------------------------------------------
pm = obj.pm;
update = ip.Results.update;
var_old = ip.Results.var_old;
coord = ip.Results.coord;
face = ip.Results.face;
init_perturb=ip.Results.init_perturb;
k_perturb=ip.Results.k_perturb;
%================================================================================
var = struct(...
    'coord', [],...
    'n_coord', [],...
    'dens', [],...
    'val', [],...
    'face', [],...
    'face_all',[],...
    'face_unq',[],...
    'n_face',[],...
    'faces_T', [],...
    'u_face',[],...
    'J',[],...
    'j_T',[],...
    'j_idx',[],...
    'j_edg',[],...
    'T_s',[],...
    'T_e',[],...
    'n_node',[],...
    'col',[],...
    'edge', [],...
    'edge_all', [],...
    'n_edg', [],...
    'f', [],...
    'dt', [],...
    'id_on_edg', [],...
    'n_on_edg', [],...
    'id_on_coord', [],...
    'n_on_coord', [],...
    'id_on_face', [],...
    'n_on_face', [],...
    'id_bound', [],...
    'n_bound', [],...
    'idMesh',[]...
     );
 if update == false
     if ~isempty(coord) && ~isempty(face)
         var.coord = coord;
         face_tem = face;
     else
         error('coord/face input required');
     end
     face_tem = sort(face_tem,2);
     var.dt = pm.dt*ones(pm.nt,1);
     var.dens=ones(size(var.coord,1),1);
 else
     var.coord = var_old.coord; face_tem = var_old.face_unq;
     var.dt = var_old.dt;
     var.dens=dens;
 end
 
 var.n_coord = size(var.coord,1);
 if init_perturb==true
     var.coord=var.coord+rand(var.n_coord,3)*obj.pm.l0*k_perturb;
 end
 var.face = nan(pm.n_val_max,3,var.n_coord);
 var.col = ones(var.n_coord,3);
var.edge = nan(pm.n_val_max*2,2,var.n_coord);
var.val = zeros(var.n_coord,1);
var.f = struct('f_edg',[],'f_ext',[],'f_P',[],'f_B',[],'f',[],'f_p',[],'f_AV',[],'cos_min',[],'kH',[],'A',[],'H',[],'d_H',[],'u_K',[]);  
var.f.A = zeros(var.n_coord,1); 
var.f.K = zeros(var.n_coord,3);
var.f.kH = zeros(var.n_coord,1);
%var.f.H = zeros(var.n_coord,1);
var.f.f = zeros(var.n_coord,3);
var.f.dH = zeros(var.n_coord,3);
%----------------------------------------------------------------------------------------
for i = 1:var.n_coord
    id_tem = face_tem == i;
    id_tem = sum(id_tem,2);
    var.val(i) = sum(id_tem);
    var.face(1:var.val(i),:,i) = face_tem(logical(id_tem),:);
%     var.edge(1:var.val(i)*2,:,i) =...
%     unique(sort([var.face(1:var.val(i),1,i),var.face(1:var.val(i),2,i);...
%                  var.face(1:var.val(i),2,i),var.face(1:var.val(i),3,i);...
%                  var.face(1:var.val(i),1,i),var.face(1:var.val(i),3,i)],2),'rows');
end
    
%================================================================================
var.face_all = []; 
for i = 1:var.n_coord
    var.face_all = [var.face_all; var.face(1:var.val(i),:,i)];
end
var.face_unq = unique(sort(var.face_all,2),'row');
var.n_face=size(var.face_unq,1);
var.edge_all=unique(sort([var.face_unq(:,1),var.face_unq(:,2);...
                          var.face_unq(:,2),var.face_unq(:,3);...
                          var.face_unq(:,1),var.face_unq(:,3)],2),'rows');

var.n_edg = size(var.edge_all,1);
if update == false
    c = var.coord(var.edge_all(:,1),:)-var.coord(var.edge_all(:,2),:);
    c_n = sqrt(sum(c.*c,2));
    l_factor = pm.l0/mean(c_n);
    var.coord = var.coord*l_factor;
end
%================================================================================
[var.j_T] = comp_j_T(var,pm);
var.j_idx = cell(var.n_coord,1);
var.faces_T = cell(var.n_coord,1);
for i = 1:var.n_coord
    id_temp = (var.face_unq(:,:) == i) ;
    id_temp = sum(id_temp,2) > 0;
    var.faces_T{i} = var.face_unq(id_temp,:);
    var.j_idx{i} = 1:var.val(i)+1;
    var.j_idx{i}(end) = 1;
end
%================================================================================
var.J = [];
var.T_s = size(var.n_coord,1);
var.T_e = size(var.n_coord,1);
for i = 1:var.n_coord
    id_temp = ~isnan(var.j_T(i,:));
    var.T_s(i) = size(var.J,2)+1;
    var.J = [var.J,var.j_T(i,id_temp)];
    var.T_e(i) = size(var.J,2);
end
var.J = var.J';
var.n_node = var.T_e-var.T_s+1;
%==========================================================================
var.u1 = cell(var.n_coord,1);var.u2 = cell(var.n_coord,1);
for i = 1:var.n_coord
    var.u1{i} = eye(var.n_node(i));
    var.u2{i} = circshift(var.u1{i},var.n_node(i)-1);
end
%==========================================================================
if obj.pm.close_surf==true
var.id_on_edg=(1:var.n_edg)';
var.id_on_coord=(1:var.n_coord)';
var.n_on_coord=numel(var.id_on_coord);
var.n_on_edg=numel(var.id_on_edg);
var.id_on_face=(1:var.n_face)';
var.n_on_face=numel(var.id_on_face)';
else
    if update == false
        var.id_bound=comp_bound(var);
        var.n_bound=numel(var.id_bound);
    else
        var.id_bound=var_old.id_bound;
        var.n_bound=var_old.n_bound;
    end
    id_tem=zeros(size(var.edge_all));
    for i=1:var.n_bound
        id_tem(var.edge_all==var.id_bound(i))=1;
    end
    id_tem=sum(id_tem,2)==2;
    var.id_on_edg=(1:var.n_edg)';
    var.id_on_edg(id_tem)=[];
    var.n_on_edg=numel(var.id_on_edg);
    var.id_on_coord=(1:var.n_coord)';
    var.id_on_coord(var.id_bound)=[];
    var.n_on_coord=numel(var.id_on_coord);
end
obj.var=var;
[obj] = getUface(obj);
end
%==========================================================================
function [j_T] = comp_j_T(var,pm)
j_T = nan(var.n_coord,pm.n_val_max);
for i = 1:var.n_coord
    ic=i;%ic = mode(var.face(:,:,i),'all');
    face_tem = var.face(1,:,i);
    face_tem = face_tem(face_tem ~= ic);
    j_T(i,1) = face_tem(1);
    j_T(i,var.val(i)) = face_tem(2);
    face_tem = var.face(:,:,i);
    for j = 2:var.val(i)-1
        j_pre = j-2; if j_pre < 1; j_pre = j_pre+var.val(i); end
        id_tem = sum((face_tem == j_T(i,j-1)),2)+sum((face_tem ~= j_T(i,j_pre)),2);
        j_T_tem = face_tem(id_tem==4,:);
        j_T_tem=j_T_tem((j_T_tem~=ic)&(j_T_tem~=j_T(i,j-1)));
        if ~isempty(j_T_tem)
            j_T(i,j) = j_T_tem((j_T_tem~=ic)&(j_T_tem~=j_T(i,j-1)));
        end
    end
end
end
%==========================================================================
function [bound] = comp_bound(var)
bound=false(var.n_coord,1);
   for i = 1:var.n_coord
       element = unique(var.face(:,:,i));
       element((isnan(element))|(element==i))=[];
       for j=1:numel(element)
           face_tem=var.face(:,:,i);
           if numel(face_tem(face_tem==element(j))) <2
               bound(i)=true;
               break;
           end
       end
   end
   id_all=(1:var.n_coord)';
   bound=id_all(bound);
end
%==========================================================================
