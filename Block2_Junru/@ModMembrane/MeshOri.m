function [obj] = MeshOri(obj, face_start, varargin)

ip = inputParser;
ip.CaseSensitive = false;
ip.addRequired('obj', @(x) isobject(x));
ip.addRequired('face_start', @(x) isnumeric(x));
ip.parse(obj,face_start,varargin{:});
%----------------------------------------------------------------------------------------
var = obj.var;
%----------------------------------------------------------------------------------------
%%
n_f = size(var.face_unq,1);
id_all = 1:n_f;
id_tem = sum((face_start(1)-var.face_unq)==0,2)+sum((face_start(2)-var.face_unq)==0,2)+sum((face_start(3)-var.face_unq)==0,2);
i_start = id_all(id_tem==3);
var.face_unq(i_start,:) = face_start;
id_par = false(n_f,1);
id_par(i_start) = true;
id_checked = false(n_f,1);
id_choose = id_all(~id_checked & id_par);

while ~isempty(id_choose)
    if max(size(id_choose)) == 1
        i = id_choose;
    else
        i=randsample(id_choose,1);
    end
res1 = var.face_unq(i,:)-var.face_unq;
res2 = [var.face_unq(i,2) var.face_unq(i,3) var.face_unq(i,1)]-var.face_unq;
res3 = [var.face_unq(i,3) var.face_unq(i,1) var.face_unq(i,2)]-var.face_unq;
id_tem1 = sum(res1==0,2);
id_tem1 = id_tem1 == 2;
id_tem2 = sum(res2==0,2);
id_tem2 = id_tem2 == 2;
id_tem3 = sum(res3==0,2);
id_tem3 = id_tem3 == 2;
%[var.face_unq(i,:);var.face_unq(id_tem1|id_tem2|id_tem3,:)]
var.face_unq(id_tem1|id_tem2|id_tem3,:) = [var.face_unq(id_tem1|id_tem2|id_tem3,3),var.face_unq(id_tem1|id_tem2|id_tem3,2),var.face_unq(id_tem1|id_tem2|id_tem3,1)];
id_tem = sum((var.face_unq(i,1)-var.face_unq)==0,2)+sum((var.face_unq(i,2)-var.face_unq)==0,2)+sum((var.face_unq(i,3)-var.face_unq)==0,2);
id_tem = id_tem==2;
id_par(id_tem) = true;
id_checked(i) = true;
id_choose = id_all(~id_checked & id_par);
end
obj.var = var;
