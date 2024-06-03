classdef ComUnit  
    properties
        typ
        unit
        unit_nat 
        unit_nat_any
    end
%==========================================================================    
    methods
%==========================================================================
        function obj = ComUnit(typ,length,T,energy,varargin)  
            ip = inputParser;
            ip.CaseSensitive = false;
            ip.addRequired('typ', @(x) ischar(x));
            ip.addRequired('length', @(x) isnumeric(x));
            ip.addRequired('T', @(x) isnumeric(x));
            ip.addRequired('energy', @(x) isnumeric(x));
            ip.parse(typ,length,T,energy,varargin{:});
%--------------------------------------------------------------------------            
            obj.typ=typ;
            if strcmp(typ,'erg')
                obj.unit=struct('length',length,'T',T,'energy',energy,'kBT',T/300*ComUnit.kBT_erg(300));
            else
                error('only erg is supported');
            end
        end
%==========================================================================        
        function obj = convert(obj,varargin)    
            ip = inputParser;
            ip.CaseSensitive = false;
            ip.addRequired('obj', @(x) isa(x,'ComUnit'));
            ip.addParameter('length', 1, @isnumeric);
            ip.addParameter('T', 300, @isnumeric);
            ip.addParameter('energy', 1, @isnumeric);
            ip.parse(obj,varargin{:});
%--------------------------------------------------------------------------
            length=ip.Results.length;
            T=ip.Results.T;
            energy=ip.Results.energy;
%--------------------------------------------------------------------------
            if strcmp(obj.typ,'erg')
                obj.unit_nat=struct('length',length/obj.unit.length,'T',T/obj.unit.T,'energy',energy/obj.unit.energy);
            end
        end
%==========================================================================        
        function [obj] = convert_high_order(obj,q,ord_length,ord_energy)    
            if strcmp(obj.typ,'erg')
                obj.unit_nat_any=q./(obj.unit.length^ord_length)...
                                  ./(obj.unit.energy^ord_energy);
            end
        end
    end
%==========================================================================    
    methods(Static)
        function x_in_cm=nm_to_cm(x_in_nm)
           x_in_cm=x_in_nm*1e-7;
        end
        function kBT_erg=kBT_erg(T) 
           kB=1.380649e-16; %kB=1.380649e-16 (in erg/K)
           kBT_erg=kB*T;
        end
        function Energy_in_erg=kBT_to_erg(Energy_in_kBT,T)  %T must be 300K
           Energy_in_erg=Energy_in_kBT*ComUnit.kBT_erg(T);
        end
    end
end