function [ output_args ] = plot_autocorr_originalonly( atc_um_filename, atc_rb2_filename, time_cts )
%COMPUTE_AUTOCORR Summary of this function goes here
%   Detailed explanation goes here

% Loading mat files
hlocs_stat_ultra = load(atc_um_filename);
hlocs_stat_ultra = hlocs_stat_ultra.hlocs_stat_um;
hlocs_stat_rb = load(atc_rb2_filename);
hlocs_stat_rb = hlocs_stat_rb.hlocs_stat_rb;

axes_fontsize = 26;
linewidth = 4;

ultra_color = [23, 87, 182]/255;
rb2_color = [60, 190, 25]/255;

% PATCH: Rectifying sequences (TO APPLY ONLY UNTIL NEW .mat FILES ARE GENERATED)
hlocs_stat_ultra = [1 hlocs_stat_ultra];
hlocs_stat_ultra(1,2:2:end) = sqrt(hlocs_stat_ultra(1,1:2:end-1).*hlocs_stat_ultra(1,3:2:end));
hlocs_stat_rb = [1 hlocs_stat_rb];

% UM atc for original sequences

fig_um = figure(1);
set(0, 'CurrentFigure', fig_um);
set(fig_um, 'DefaultAxesFontName', 'Times New Roman');
set(fig_um, 'DefaultTextFontName', 'Times New Roman');
clf;
hold on;

for tc_id = 1:size(time_cts,2)
    h = line([time_cts(tc_id) time_cts(tc_id)],[1E-2 1E0], 'Color',ultra_color, 'LineStyle','--');
    set(h,'linewidth',2);
    text(0.75*time_cts(tc_id),1.2E-2,num2str(time_cts(tc_id),2), 'Color',ultra_color, 'Rotation',90, 'FontSize',16, 'FontName', 'TimesNewRoman');
end

h=plot(hlocs_stat_ultra, 'Color',ultra_color, 'LineStyle','-');%,'DisplayName','Ultrametric - Original sequence');
set(h,'linewidth',linewidth);
    
ax = gca;
ax.FontSize = axes_fontsize; 

set(gca,'Xscale','log');
set(gca,'Yscale','log');

xlabel("\Delta t", 'FontSize', axes_fontsize);
ylabel("P_{0}(\Delta t)", 'FontSize', axes_fontsize);

% RB2 atc for original sequences

fig_rb2 = figure(2);
set(0, 'CurrentFigure', fig_rb2);
set(fig_rb2, 'DefaultAxesFontName', 'Times New Roman');
set(fig_rb2, 'DefaultTextFontName', 'Times New Roman');
clf;
hold on;

h=plot(hlocs_stat_rb, 'Color',rb2_color, 'LineStyle','-');
set(h,'linewidth',linewidth);
set(gca, 'FontName', 'TimesNewRoman');
    
ax = gca;
ax.FontSize = axes_fontsize; 

set(gca,'Xscale','log', 'FontName', 'TimesNewRoman');
set(gca,'Yscale','log', 'FontName', 'TimesNewRoman');

xlabel("\Delta t", 'FontSize', axes_fontsize);
ylabel("P_{0}(\Delta t)", 'FontSize', axes_fontsize);

% Put UM and RB2 atc for original sequences on a same figure

fig_both = figure(3);
set(0, 'CurrentFigure', fig_both);
set(fig_both, 'DefaultAxesFontName', 'Times New Roman');
set(fig_both, 'DefaultTextFontName', 'Times New Roman');
clf;
hold on;

for tc_id = 1:size(time_cts,2)
    h = line([time_cts(tc_id) time_cts(tc_id)],[1E-2 1E0], 'Color',ultra_color, 'LineStyle','--');
    set(h,'linewidth',2);
    text(0.75*time_cts(tc_id),1.2E-2,num2str(time_cts(tc_id),2), 'Color',ultra_color, 'Rotation',90, 'FontSize',16, 'FontName', 'TimesNewRoman');
end

h=plot(hlocs_stat_rb,'Color',rb2_color,'LineStyle','-');
set(h,'linewidth',linewidth);
h=plot(hlocs_stat_ultra,'Color',ultra_color,'LineStyle','-');%,'DisplayName','Ultrametric - Original sequence');
set(h,'linewidth',linewidth);

ax = gca;
ax.FontSize = axes_fontsize; 

set(gca,'Xscale','log', 'FontName', 'TimesNewRoman');
set(gca,'Yscale','log', 'FontName', 'TimesNewRoman');

xlabel("\Delta t", 'FontSize', axes_fontsize);
ylabel("P_{0}(\Delta t)", 'FontSize', axes_fontsize);

%text(pos(1,1)+0.5*pos(1,3)+0.08, 1.12*(pos(1,2)+pos(1,4)), 'Random blocks sequence', 'Units', 'normalized', 'FontSize', 20, 'FontWeight', 'bold');

end

