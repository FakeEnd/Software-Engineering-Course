function [f] = plot(obj,varargin)
ip = inputParser;
ip.addRequired('obj', @(x) isobject(x));
ip.addParameter('f', [], @isobject);
ip.addParameter('col', [], @isnumeric);
ip.addParameter('col_min', 0, @isnumeric);
ip.addParameter('col_max', 1, @isnumeric);
ip.addParameter('FaceAlpha', 1, @isnumeric);
ip.addParameter('bar_name', {'\kappa'}, @iscell);
ip.addParameter('PaperPosition', [0 0 5 5], @isnumeric);
ip.addParameter('LineStyle', '-', @ischar);
ip.addParameter('light_ang',[0 30]);
ip.addParameter('LineWidth', 1, @isnumeric);
ip.addParameter('view_ang', [10 30], @isnumeric);
ip.addParameter('plot_adp', false, @islogical);
ip.addParameter('plot_sm_can', false, @islogical);
ip.addParameter('split', [], @isstruct);
ip.addParameter('merge', [], @isstruct);
ip.addParameter('colBar', false, @islogical);
ip.parse(obj, varargin{:});
%==========================================================================
vertices = obj.var.coord;
n_ver = size(vertices,1);
faces = obj.var.face_unq;
f = ip.Results.f;
col = ip.Results.col;
if isempty(col)
    col = ones(n_ver,1);
end
bar_name = ip.Results.bar_name;
PaperPosition = ip.Results.PaperPosition;
LineStyle = ip.Results.LineStyle;
col_min = ip.Results.col_min;
col_max = ip.Results.col_max;
LineWidth = ip.Results.LineWidth;
view_ang=ip.Results.view_ang;
light_ang=ip.Results.light_ang;
colBar=ip.Results.colBar;
%==========================================================================
if isempty(f)
    f=figure;
else
    figure(f); hold on;
end
n_cmap = 1000;
n_col = max(size(col));
cmap = cool(n_cmap);
colormap(cmap);
col_map = zeros(n_col,3);
for i = 1:max(size(col))
    [~,id_col] = min(abs(col(i)-(col_min+(1:n_cmap)/n_cmap)*(col_max-col_min)));
    col_map(i,:) = cmap(id_col,:);
end
% col_map = col_min+col_map*(col_max-col_min);

p = patch('Vertices',vertices,'Faces',faces,'FaceVertexCData',col_map,'FaceColor','interp','facealpha',1);
if colBar==true
handleToColorBar = colorbar;
handleToColorBar.Limits=[0 1];
% set(handleToColorBar,'Ticks', (col_min:(col_max-col_min)*0.2:col_max));
% n_tick = size(handleToColorBar.Ticks,2);
% TickLabels = cell(n_tick,1);
% for i = 1:n_tick
%     TickLabels{i} = num2str(handleToColorBar.Ticks(i));
% end
set(handleToColorBar,'TickLabels', []);
set(handleToColorBar,'Ticks',[])

% x_tem = get(handleToColorBar,'Position');
% set(handleToColorBar,'Position', [x_tem(1) x_tem(2)+0.01 x_tem(3) x_tem(4)]);
%set(handleToColorBar,'YTickLabel', []);
%hYLabel = ylabel(handleToColorBar, 'Blue      Cyan      Green      Yellow      Orange      Red');
%set(hYLabel,'Rotation',90);
title(handleToColorBar, bar_name,'FontWeight','bold','FontSize',12);
end
%view(3); grid on; axis square; title('Cylinder','FontSize',12)
p.FaceAlpha = ip.Results.FaceAlpha;           % remove the transparency
%p.FaceColor = 'interp';    % set the face colors to be interpolated
p.LineStyle = LineStyle; %p.LineStyle = 'none';      % remove the lines
p.LineWidth = LineWidth;
%colormap(copper)
f.PaperUnits = 'centimeters';
f.PaperPosition = PaperPosition;
%--------------------------------------------------------------------------
if ip.Results.plot_adp==true
figure(f); hold on;
scatter3(obj.var.coord(obj.var.id_adp,1),obj.var.coord(obj.var.id_adp,2),obj.var.coord(obj.var.id_adp,3),10,'filled','MarkerFaceColor',[0 1 0.5]);
end
%--------------------------------------------------------------------------
if (obj.pm.close_surf == false) %|| (relaxed == false)
    %save('temp.mat','var','edg_add');
   figure(f); hold on;
   scatter3(obj.var.coord(obj.var.id_bound,1),obj.var.coord(obj.var.id_bound,2),obj.var.coord(obj.var.id_bound,3),40,'filled','MarkerFaceColor',[0 1 1]); hold on;
   n_ring_edg=numel(obj.var.id_on_edg);
   figure(f); hold on;
% for i = 1:n_ring_edg    
%     plot3([obj.var.coord(obj.var.edge_all(obj.var.id_on_edg(i),1),1),obj.var.coord(obj.var.edge_all(obj.var.id_on_edg(i),2),1)],...
%           [obj.var.coord(obj.var.edge_all(obj.var.id_on_edg(i),1),2),obj.var.coord(obj.var.edge_all(obj.var.id_on_edg(i),2),2)],...
%           [obj.var.coord(obj.var.edge_all(obj.var.id_on_edg(i),1),3),obj.var.coord(obj.var.edge_all(obj.var.id_on_edg(i),2),3)],'linewidth',1,'color',[1 0 0]);
% end
end
%--------------------------------------------------------------------------
       %%
if ip.Results.plot_sm_can==true  
    split=ip.Results.split;
    merge=ip.Results.merge;
    id_edg_all=(1:obj.var.n_edg)';
    i_no_split=id_edg_all(~split.can);
    n_no_split=numel(i_no_split);
    i_no_merge=id_edg_all(~merge.can);
    n_no_merge=numel(i_no_merge);
        for i=1:n_no_split
            hold on;
            plot3([obj.var.coord(obj.var.edge_all(i_no_split(i),1),1),obj.var.coord(obj.var.edge_all(i_no_split(i),2),1)],...
                  [obj.var.coord(obj.var.edge_all(i_no_split(i),1),2),obj.var.coord(obj.var.edge_all(i_no_split(i),2),2)],...
                  [obj.var.coord(obj.var.edge_all(i_no_split(i),1),3),obj.var.coord(obj.var.edge_all(i_no_split(i),2),3)],'linewidth',2,'color',[1 0 0]);
        end
        for i=1:n_no_merge
            hold on;
            plot3([obj.var.coord(obj.var.edge_all(i_no_merge(i),1),1),obj.var.coord(obj.var.edge_all(i_no_merge(i),2),1)],...
                  [obj.var.coord(obj.var.edge_all(i_no_merge(i),1),2),obj.var.coord(obj.var.edge_all(i_no_merge(i),2),2)],...
                  [obj.var.coord(obj.var.edge_all(i_no_merge(i),1),3),obj.var.coord(obj.var.edge_all(i_no_merge(i),2),3)],'--','linewidth',2,'color',[0 1 0]);
        end
end
%--------------------------------------------------------------------------
l=light('Style','infinite');
lightangle(l,light_ang(1),light_ang(2))
material dull 
view(view_ang(1),view_ang(2));
camlight('headlight');
l=light('Style','infinite');
lightangle(l,light_ang(1),light_ang(2))
material dull 
view(view_ang(1),view_ang(2));
camlight('headlight');
%--------------------------------------------------------------------------
end