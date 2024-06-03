
function [m] = getUface(m, varargin)

ip = inputParser;
ip.CaseSensitive = false;
ip.addRequired('m', @(x) isobject(x));
ip.parse(m, varargin{:});
%--------------------------------------------------------------------------------------------------------
n_face=size(m.var.face_unq,1);
id_all=(1:n_face)';
m.var.u_face=zeros(m.var.n_coord,3);
for i_on = 1:m.var.n_on_coord
    i=m.var.id_on_coord(i_on);
    face_start=[i, m.var.j_T(i,1), m.var.j_T(i,2)];
    id_tem=(m.var.face_unq==face_start(1))+(m.var.face_unq==face_start(2))+(m.var.face_unq==face_start(3));
    id_tem=sum(id_tem,2);
    j_save=id_all(id_tem==3);
    a=[m.var.coord(m.var.face_unq(j_save,2),1)-m.var.coord(m.var.face_unq(j_save,1),1),...
       m.var.coord(m.var.face_unq(j_save,2),2)-m.var.coord(m.var.face_unq(j_save,1),2),...
       m.var.coord(m.var.face_unq(j_save,2),3)-m.var.coord(m.var.face_unq(j_save,1),3)]; 
    b=[m.var.coord(m.var.face_unq(j_save,3),1)-m.var.coord(m.var.face_unq(j_save,2),1),...
       m.var.coord(m.var.face_unq(j_save,3),2)-m.var.coord(m.var.face_unq(j_save,2),2),...
       m.var.coord(m.var.face_unq(j_save,3),3)-m.var.coord(m.var.face_unq(j_save,2),3)]; 
    m.var.u_face(i,:)=[a(2)*b(3)-a(3)*b(2),a(3)*b(1)-a(1)*b(3),a(1)*b(2)-a(2)*b(1)];
end
end
%==========================================================================
