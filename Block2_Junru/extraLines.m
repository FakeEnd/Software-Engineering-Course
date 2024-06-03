% 0. setting internal force's strength using recommended value:
m.pm.Vdh.V0=0.1;
% 1. getting internal force:
Fi = Finternal(m,'plot_or_not',false);
% 2. spatial range for remesh later
rLim=[min(Fi.rg),max(Fi.rg)]; 
% 3. getting the adaptive time step, 
% input: Fb,Fv,Fs: bending, volume, surface force, mu: mobility in Langevin eq.
% output: dt-adaptive time step, Ftot: total force, l: lengths of edges
% use Ftot for future computations
[dt,Ftot,l]=varDt(m,Fi,Fb+Fv+Fs,mu);
% 4. remeshing
% input: rLim from step2, l from step3, Fi from step1
% output: new membrane with better triangles,~ means nonessential output
[m,~] = RemeshCtrl(m,Fi,rLim,'l0',l,'print_or_not', false);

% other parameter recommendations:
mu=100;
% explore kappa for Fb, kv for Fv, ks for Fs