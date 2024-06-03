classdef ModMembrane
%==========================================================================
%==========================================================================    
    properties
       var
       pm
       prop
    end
%==========================================================================
%==========================================================================   
    methods
        function obj = ModMembrane(n_ico_sphere,varargin)
%--------------------------------------------------------------------------
            ip = inputParser;
            ip.CaseSensitive = false;
            ip.addRequired('n_ico_sphere', @(x) isnumeric(x));
            ip.addParameter('unit', [], @isobject);
            ip.addParameter('update', false, @islogical);
            ip.addParameter('pm_exist', [], @isstruct);
            ip.parse(n_ico_sphere,varargin{:}); 
%--------------------------------------------------------------------------
            unit=ip.Results.unit;
            if isempty(unit)
                unit=nat_unit('erg',ComUnit.nm_to_cm(1),300);
                warning('no unit assigned, using 1nm and 300K as natural units')
            end
%--------------------------------------------------------------------------
            obj.prop = 'membrane';
            [obj] = SetParameter(obj,'n_ico_sphere',n_ico_sphere,'unit',unit);

            [ver_tem,face_tem] = icosphere(obj);
            [obj] = SetVar(obj,'coord',ver_tem,'face',face_tem);
            [obj] = MeshOri(obj, obj.var.face_unq(1,:));
        end
%==========================================================================        
        function A = Area(obj)                     
            pmc=zeros(10,1);
            pmc(1) = obj.pm.nt;
            pmc(2) = obj.pm.dt;
            pmc(3) = obj.pm.P;
            pmc(4) = obj.pm.k_c;
            pmc(5) = obj.pm.k_e;
            pmc(6) = obj.pm.dr;
            pmc(7) = obj.pm.k_V;
            pmc(8) = obj.pm.k_A;
            pmc(9) = obj.pm.V0;
            pmc(10) = obj.pm.A0;
            pmc(11) = obj.pm.nAVmean;
            pmc(12) = obj.pm.k_a;
            
            j_T = obj.var.j_T; j_T(isnan(j_T)) = 0;
            
            [A]=obj.AreaMex...
                (obj.var.coord,...
                pmc,...
                obj.var.edge_all,...
                obj.var.face_unq,...
                j_T,...
                obj.var.id_on_coord,...
                obj.var.n_node',...
                obj.var.T_s',...
                obj.var.T_e');
        end
%==========================================================================        
        function V = Volume(obj)                     
            pmc=zeros(10,1);
            pmc(1) = obj.pm.nt;
            pmc(2) = obj.pm.dt;
            pmc(3) = obj.pm.P;
            pmc(4) = obj.pm.k_c;
            pmc(5) = obj.pm.k_e;
            pmc(6) = obj.pm.dr;
            pmc(7) = obj.pm.k_V;
            pmc(8) = obj.pm.k_A;
            pmc(9) = obj.pm.V0;
            pmc(10) = obj.pm.A0;
            pmc(11) = obj.pm.nAVmean;
            pmc(12) = obj.pm.k_a;
            
            j_T = obj.var.j_T; j_T(isnan(j_T)) = 0;
            
            [V]=obj.VolumeMex...
                (obj.var.coord,...
                pmc,...
                obj.var.edge_all,...
                obj.var.face_unq,...
                j_T,...
                obj.var.id_on_face,...
                obj.var.n_node',...
                obj.var.T_s',...
                obj.var.T_e');
        end        
%==========================================================================        
function H = Helfrich(obj)                     
            pmc=zeros(10,1);
            pmc(1) = obj.pm.nt;
            pmc(2) = obj.pm.dt;
            pmc(3) = obj.pm.P;
            pmc(4) = obj.pm.k_c;
            pmc(5) = obj.pm.k_e;
            pmc(6) = obj.pm.dr;
            pmc(7) = obj.pm.k_V;
            pmc(8) = obj.pm.k_A;
            pmc(9) = obj.pm.V0;
            pmc(10) = obj.pm.A0;
            pmc(11) = obj.pm.nAVmean;
            pmc(12) = obj.pm.k_a;
            
            j_T = obj.var.j_T; j_T(isnan(j_T)) = 0;
            
            [H]=obj.HelfrichMex...
                (obj.var.coord,...
                pmc,...
                obj.var.edge_all,...
                obj.var.face_unq,...
                j_T,...
                obj.var.id_on_coord,...
                obj.var.n_node',...
                obj.var.T_s',...
                obj.var.T_e');
        end        
%==========================================================================  
        [f] = plot(obj,varargin);
        [obj,remeshed,edg_add] = Remesh(obj,rmTyp,rmIdx,rLim,varargin);
        [var] = AddVertex(obj,pm,var,ver_new,edge_all_new,face_new, varargin);
        [obj] = getUface(obj, varargin);
        [id_ring_ver1,id_ring_edg1,id_ring_ver2,id_ring_edg2] = Ring(obj,edge_ctr, varargin);
        [Fi] = Finternal(m, varargin);
    end
%==========================================================================
%==========================================================================    
methods(Static)
    function mod_name = identify(varargin)
        mod_name = 'ModMembrane';
    end
    [A]=AreaMex(coord,pmc,edge_all,face_unq,j_T,id_on_coord,n_node,T_s,T_e);
    [V]=VolumeMex(coord,pmc,edge_all,face_unq,j_T,id_on_coord,n_node,T_s,T_e);
    [H]=HelfrichMex(coord,pmc,edge_all,face_unq,j_T,id_on_coord,n_node,T_s,T_e);
    [V] = Vinternal(r,Vpm,Vcase);
end
end