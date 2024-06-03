function [dt,Ftot,l]=varDt(m,Fi,Frest,mu)
%%
dr=m.var.coord(m.var.edge_all(:,2),:)-m.var.coord(m.var.edge_all(:,1),:);
l=sqrt(sum(dr.^2,2));
d=l-Fi.rg';
u=dr./l;
[~,id]=min(abs(d),[],2);
idTem=Fi.rg(id)-l<0;
id(idTem)=id(idTem)+1;

drFi=Fi.rn(2)-Fi.rn(1);
ir=floor(l/drFi+0.5);
Ftem=Fi.fn(ir);
Fint=zeros(m.var.n_coord,3);

for i=1:m.var.n_coord
    idTem=m.var.edge_all(:,1)==i;
    Fint(i,:)= Fint(i,:)-sum(Ftem(idTem).*u(idTem,:));
    idTem=m.var.edge_all(:,2)==i;
    Fint(i,:)= Fint(i,:)+sum(Ftem(idTem).*u(idTem,:));
end

Ftot=Frest+Fint;
df=Ftot(m.var.edge_all(:,2),:)-Ftot(m.var.edge_all(:,1),:);

lr=0.5*(Fi.rg(id)+Fi.rg(id+1));
ll=0.5*(Fi.rg(id-1)+Fi.rg(id-2));
c=[ll.^2-l.^2,lr.^2-l.^2];
b=2*mu*sum(dr.*df,2);
dtTem=c./b;
dtTem(dtTem<0)=[];
dt=min(dtTem);
end