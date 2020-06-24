function [ output_args ] = plot_autocorr( atc_um_filename, atc_rb2_filename, block_sizes, leg_x, leg_y )
%COMPUTE_AUTOCORR Summary of this function goes here
%   Detailed explanation goes here


% Loading mat files
hlocs_stat_ultra = load(atc_um_filename);
hlocs_stat_ultra = hlocs_stat_ultra.hlocs_stat_ultra;
hlocs_stat_rb = load(atc_rb2_filename);
hlocs_stat_rb = hlocs_stat_rb.hlocs_stat_rb;

% ------------------------ %
% 1) Ultrametric sequences %
% ------------------------ %
figure(1);
clf;
hold on;

ultra_cb = [1, 0, 0];
ultra_ce = [230, 130, 130]/255;
ultra_colors = [linspace(ultra_cb(1),ultra_ce(1),1+length(block_sizes))', linspace(ultra_cb(2),ultra_ce(2),1+length(block_sizes))', linspace(ultra_cb(3),ultra_ce(3),1+length(block_sizes))'];

%legends = char.empty;
legends = {};

h=plot(hlocs_stat_ultra(1,2:2:end-1)./hlocs_stat_ultra(1,2),'Color',ultra_colors(1,:),'LineStyle','-');%,'DisplayName','Ultrametric - Original sequence');
legends{1} = 'Original sequence';
set(h,'linewidth',2);

for block_sz_id = 1:length(block_sizes)
    fig_name = strcat(sprintf('Shuffled, block size %s', int2str(block_sizes(block_sz_id))));
    h=plot(hlocs_stat_ultra(1+block_sz_id,2:2:end-1)./hlocs_stat_ultra(1+block_sz_id,2),'Color',ultra_colors(1+block_sz_id,:),'LineStyle','-');%,'DisplayName',fig_name);
    legends{1+block_sz_id} = fig_name;
    set(h,'linewidth',2);
end

% Figure configuration
ax = gca;
ax.FontSize = 18; 

set(gca,'Xscale','log');
set(gca,'Yscale','log');

xlabel("\Delta t", 'FontSize', 18);
ylabel("P_{0}(\Delta t)", 'FontSize', 18);

columnlegend(2, legends, 'FontSize', 18, 'padding', 0.6);
hLegend = findobj(gcf, 'Type', 'Legend');
pos = get(hLegend,'position');
pos(1,1) = leg_x;
pos(1,2) = leg_y;
set(hLegend,'position',pos);

%text(pos(1,1), 1.12*(pos(1,2)+pos(1,4)), 'Ultrametric sequence', 'Units', 'normalized', 'FontSize', 20, 'FontWeight', 'bold');

% ------------------------- %
% 2) Random block sequences %
% ------------------------- %

figure(2);
clf;
hold on;

rb_cb = [0, 110, 0]/255;
rb_ce = [145, 225, 145]/255;
rb_colors = [linspace(rb_cb(1),rb_ce(1),1+length(block_sizes))', linspace(rb_cb(2),rb_ce(2),1+length(block_sizes))', linspace(rb_cb(3),rb_ce(3),1+length(block_sizes))'];

h=plot(hlocs_stat_rb(1,2:2:end-1)./hlocs_stat_rb(1,2),'Color',rb_colors(1,:),'LineStyle','-');%,'DisplayName','Random blocks - Original sequence');
legends{2+length(block_sizes)} = 'Original sequence';
set(h,'linewidth',2);

for block_sz_id = 1:length(block_sizes)
    fig_name = strcat(sprintf('Shuffled, block size %s', int2str(block_sizes(block_sz_id))));
    h=plot(hlocs_stat_rb(1+block_sz_id,2:2:end-1)./hlocs_stat_rb(1+block_sz_id,2),'Color',rb_colors(1+block_sz_id,:),'LineStyle','--');%,'DisplayName',fig_name);
    legends{2+length(block_sizes)+block_sz_id} = fig_name;
    set(h,'linewidth',2);
end

% Figure configuration
ax = gca;
ax.FontSize = 18; 

set(gca,'Xscale','log');
set(gca,'Yscale','log');

xlabel("Sequence space", 'FontSize', 18);
ylabel("Autocorrelation", 'FontSize', 18);

columnlegend(2, legends, 'FontSize', 18, 'padding', 0.6);
hLegend = findobj(gcf, 'Type', 'Legend');
pos = get(hLegend,'position');
pos(1,1) = leg_x;
pos(1,2) = leg_y;
set(hLegend,'position',pos);

%text(pos(1,1)+0.5*pos(1,3)+0.08, 1.12*(pos(1,2)+pos(1,4)), 'Random blocks sequence', 'Units', 'normalized', 'FontSize', 20, 'FontWeight', 'bold');

end

