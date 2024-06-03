function [V] = Vinternal(r,Vpm,Vcase)

ip = inputParser;
ip.CaseSensitive = false;
ip.addRequired('r', @(x) isnumeric);
ip.addRequired('Vpm', @(x) isstruct);
ip.addRequired('Vcase', @(x) isnumeric); % same as m.pm.remeshScheme
%ip.parse(r,r_1,r_2,rb_11,rb_12,rb_21,rb_22,k_w,k_b11,k_b12,k_b21,k_b22,e_w,e_b1,e_b2);
%--------------------------------------------------------------------------------------------------------
if Vcase>0
    V = Vpm.V0*(1./tanh((r-Vpm.r_1)*Vpm.k_w)+1./tanh((Vpm.r_2-r)*Vpm.k_w)...
      + Vpm.e_b1/Vpm.e_w./(1+exp(-Vpm.k_b11*(r-Vpm.rb_11)))+Vpm.e_b1/Vpm.e_w./(1+exp(-Vpm.k_b12*(Vpm.rb_12-r)))...
      + Vpm.e_b2/Vpm.e_w./(1+exp(-Vpm.k_b21*(r-Vpm.rb_21)))+Vpm.e_b2/Vpm.e_w./(1+exp(-Vpm.k_b22*(Vpm.rb_22-r)))...
        );
else
    V = Vpm.V0*(1./tanh((r-Vpm.r_1)*Vpm.k_w)+1./tanh((Vpm.r_2-r)*Vpm.k_w)...
      + Vpm.e_b2/Vpm.e_w./(1+exp(-Vpm.k_b21*(r-Vpm.rb_21)))+Vpm.e_b2/Vpm.e_w./(1+exp(-Vpm.k_b22*(Vpm.rb_22-r)))...
        );
end
end
