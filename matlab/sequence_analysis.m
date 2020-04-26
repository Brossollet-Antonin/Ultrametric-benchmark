maxh=10000;
tree_levels=5;
tree_l=2^tree_levels;

block_sizes = [1, 100, 200, 400, 800, 1600, 8000];
sequences.ultra.orig = int16.empty;
sequences.ultra.shfl = containers.Map;
sequences.rb.orig = int16.empty;
sequences.rb.shfl = containers.Map;

for block_sz_id = 1:length(block_sizes)
    block_sz = int2str(block_sizes(block_sz_id));
    sequences.ultra.shfl(block_sz) = int16.empty;
    sequences.rb.shfl(block_sz) = int16.empty;
end

ultra_dataroot = 'ArticleRoot/data/artificial_32/FCL20/ultrametric_length200000_batches10_seqlen200_ratio8/matlab';
rb_dataroot = 'ArticleRoot/data/artificial_32/FCL20/random_blocks2_length200000_seqlen200_ratio8/matlab/split1600';

% Loading ultrametric sequences data
cd ~/Documents/Workspace/Jobs/Columbia/ultrametric_benchmark/Ultrametric-benchmark;
cd(ultra_dataroot)
ultra_file_list = dir('*.mat');

for file_id = 1:length(ultra_file_list)
  baseFileName = ultra_file_list(file_id).name;
  fullFileName = fullfile(ultra_dataroot, baseFileName);
  new_seq = load(fullFileName);
  for block_sz_id = 1:length(block_sizes)
      block_sz = int2str(block_sizes(block_sz_id));
      if(contains(fullFileName, strcat('_shfl',block_sz,'_')))
          sequences.ultra.shfl(block_sz) = [sequences.ultra.shfl(block_sz);new_seq.lbl_seq];
          break;
      end
  end
  
  if(contains(fullFileName, strcat('_orig_')))
      sequences.ultra.orig = [sequences.ultra.orig;new_seq.lbl_seq];
  end
end

% Loading random block sequences data
cd ~/Documents/Workspace/Jobs/Columbia/ultrametric_benchmark/Ultrametric-benchmark;
cd(rb_dataroot)
rb_file_list = dir('*.mat');

for file_id = 1:length(rb_file_list)
  baseFileName = rb_file_list(file_id).name;
  fullFileName = fullfile(rb_dataroot, baseFileName);
  new_seq = load(fullFileName);
  for block_sz_id = 1:length(block_sizes)
      block_sz = int2str(block_sizes(block_sz_id));
      if(contains(fullFileName, strcat('_shfl',block_sz,'_')))
          sequences.rb.shfl(block_sz) = [sequences.rb.shfl(block_sz);new_seq.lbl_seq];
          break;
      end
  end
  
  if(contains(fullFileName, strcat('_orig_')))
      sequences.rb.orig = [sequences.rb.orig;new_seq.lbl_seq];
  end
end

% compute P_0 for ultrametrics sequences

hlocs_stat_ultra=zeros(length(block_sizes)+1,maxh-1);

disp('Ultrametric - Computing P0 for original sequences...')
for seq_id = 1:size(sequences.ultra.orig,1)
    lbl_seq = sequences.ultra.orig(seq_id,:);
    for i=1:tree_l
        locs=find(lbl_seq==i); %find all occurrences of i
        nd=1;
        for j=1:size(locs,2)
            for k=(j+1):size(locs,2)
                locsd(nd)=locs(k)-locs(j);
                if(locsd(nd)>maxh)
                    break;
                end
                nd=nd+1;
            end
        end

        if(nd>1)
            edges=1:maxh;
            hlocs= histcounts(locsd(1:(nd-1)),edges,'Normalization','probability'); % compute the histogram of the lags
            hlocs_stat_ultra(1,:)=hlocs_stat_ultra(1,:)+hlocs./tree_l;
        end
    end
end

for block_sz_id = 1:length(block_sizes)
    block_sz = int2str(block_sizes(block_sz_id));
    fprintf('Ultrametric - Computing P0 for shuffled sequences with block size %s...\n', block_sz)
    seq_list = sequences.ultra.shfl(block_sz);
    for seq_id = 1:size(sequences.ultra.shfl(block_sz),1)
        lbl_seq = seq_list(seq_id,:);
        for i=1:tree_l
            locs=find(lbl_seq==i); %find all occurrences of i
            nd=1;
            for j=1:size(locs,2)
                for k=(j+1):size(locs,2)
                    locsd(nd)=locs(k)-locs(j);
                    if(locsd(nd)>maxh)
                        break;
                    end
                    nd=nd+1;
                end
            end

            if(nd>1)
            edges=1:maxh;
            hlocs= histcounts(locsd(1:(nd-1)),edges,'Normalization','probability'); % compute the histogram of the lags
            hlocs_stat_ultra(1+block_sz_id,:)=hlocs_stat_ultra(1+block_sz_id,:)+hlocs./tree_l;
            end
        end
    end
end


% compute P_0 for ultrametrics sequences

hlocs_stat_rb=zeros(length(block_sizes)+1,maxh-1);

disp('Random blocks - Computing P0 for original sequences...')
for seq_id = 1:size(sequences.rb.orig,1)
    lbl_seq = sequences.rb.orig(seq_id,:);
    for i=1:tree_l
        locs=find(lbl_seq==i); %find all occurrences of i
        nd=1;
        for j=1:size(locs,2)
            for k=(j+1):size(locs,2)
                locsd(nd)=locs(k)-locs(j);
                if(locsd(nd)>maxh)
                    break;
                end
                nd=nd+1;
            end
        end

        if(nd>1)
            edges=1:maxh;
            hlocs= histcounts(locsd(1:(nd-1)),edges,'Normalization','probability'); % compute the histogram of the lags
            hlocs_stat_rb(1,:)=hlocs_stat_rb(1,:)+hlocs./tree_l;
        end
    end
end

for block_sz_id = 1:length(block_sizes)
    block_sz = int2str(block_sizes(block_sz_id));
    fprintf('Random blocks - Computing P0 for shuffled sequences with block size %s...\n', block_sz)
    seq_list = sequences.rb.shfl(block_sz);
    for seq_id = 1:size(sequences.rb.shfl(block_sz),1)
        lbl_seq = seq_list(seq_id,:);
        for i=1:tree_l
            locs=find(lbl_seq==i); %find all occurrences of i
            nd=1;
            for j=1:size(locs,2)
                for k=(j+1):size(locs,2)
                    locsd(nd)=locs(k)-locs(j);
                    if(locsd(nd)>maxh)
                        break;
                    end
                    nd=nd+1;
                end
            end

            if(nd>1)
            edges=1:maxh;
            hlocs= histcounts(locsd(1:(nd-1)),edges,'Normalization','probability'); % compute the histogram of the lags
            hlocs_stat_rb(1+block_sz_id,:)=hlocs_stat_rb(1+block_sz_id,:)+hlocs./tree_l;
            end
        end
    end
end


% Plotting results
figure(1);
clf;
hold on;

ultra_cb = [1, 0, 0];
ultra_ce = [230, 130, 130]/255;
ultra_colors = [linspace(ultra_cb(1),ultra_ce(1),1+length(block_sizes))', linspace(ultra_cb(2),ultra_ce(2),1+length(block_sizes))', linspace(ultra_cb(3),ultra_ce(3),1+length(block_sizes))'];

h=plot(hlocs_stat_ultra(1,2:2:end-1)./hlocs_stat_ultra(1,2),'Color',ultra_colors(1,:),'LineStyle','-','DisplayName','Ultrametric - Original sequence');
set(h,'linewidth',2);

for block_sz_id = 1:length(block_sizes)
    fig_name = strcat('Ultrametric - Shuffled with block size ', int2str(block_sizes(block_sz_id)));
    h=plot(hlocs_stat_ultra(1+block_sz_id,2:2:end-1)./hlocs_stat_ultra(1+block_sz_id,2),'Color',ultra_colors(1+block_sz_id,:),'LineStyle','--','DisplayName',fig_name);
    set(h,'linewidth',2);
end

rb_cb = [0, 110, 0]/255;
rb_ce = [145, 225, 145]/255;
rb_colors = [linspace(rb_cb(1),rb_ce(1),1+length(block_sizes))', linspace(rb_cb(2),rb_ce(2),1+length(block_sizes))', linspace(rb_cb(3),rb_ce(3),1+length(block_sizes))'];

h=plot(hlocs_stat_rb(1,2:2:end-1)./hlocs_stat_rb(1,2),'Color',rb_colors(1,:),'LineStyle','-','DisplayName','Random blocks - Original sequence');
set(h,'linewidth',2);

for block_sz_id = 1:length(block_sizes)
    fig_name = strcat('Random blocks - Split size ', int2str(block_sizes(block_sz_id)));
    h=plot(hlocs_stat_rb(1+block_sz_id,2:2:end-1)./hlocs_stat_rb(1+block_sz_id,2),'Color',rb_colors(1+block_sz_id,:),'LineStyle','--','DisplayName',fig_name);
    set(h,'linewidth',2);
end

set(gca,'xscale','log');
set(gca,'yscale','log');


xlabel('\Delta t');
ylabel('P_0(\Delta t)');
set(gca,'fontsize',14);
legend



function [a] = loadpickle(filename)
    if ~exist(filename,'file')
        error('%s is not a file',filename);
    end
    outname = [tempname() '.mat'];
    pyscript = ['import cPickle as pickle;import sys;import scipy.io;file=open("' filename '","r");dat=pickle.load(file);file.close();scipy.io.savemat("' outname '",dat)'];
    system(['LD_LIBRARY_PATH=/opt/intel/composer_xe_2013/mkl/lib/intel64:/opt/intel/composer_xe_2013/lib/intel64;python -c ''' pyscript '''']);
    a = load(outname);
end