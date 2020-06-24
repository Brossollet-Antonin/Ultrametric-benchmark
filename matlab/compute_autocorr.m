function [ output_args ] = compute_autocorr( dataset_name, nnarchi, um_battery, rb2_battery )
%COMPUTE_AUTOCORR Summary of this function goes here
%   Detailed explanation goes here
maxh=10000;

project_root='~/ultrametric_benchmark/Ultrametric-benchmark';
data_root=fullfile(project_root,'Results/1toM',dataset_name,nnarchi);
um_root=fullfile(data_root,um_battery);
rb2_root=fullfile(data_root,rb2_battery);

% Getting parameter files
um_params = jsondecode(fileread(fullfile(um_root, 'parameters.json')));
rb2_params = jsondecode(fileread(fullfile(rb2_root, 'parameters.json')));

% Checking consistency
assert(um_params.TreeDepth==rb2_params.TreeDepth);
assert(um_params.TreeBranching==rb2_params.TreeBranching);

command=um_params.OriginalCommand;
block_sizes_raw=split(command,"'--blocksz',");
block_sizes_raw=block_sizes_raw(2,1);
block_sizes = regexp(block_sizes_raw,'\d*','Match');
block_sizes = [block_sizes{:}];
tmp_sz = numel(block_sizes);
tmp_blsz = NaN(tmp_sz,1);
for k = 1:tmp_sz
  tmp_blsz(k,1) = str2num(block_sizes{k});
end
block_sizes = tmp_blsz;

% Reading block sizes, tree size, etc
tree_levels=um_params.TreeDepth;
branching=um_params.TreeBranching;
tree_l=branching^tree_levels;

% Initializing objects
sequences.ultra.orig = int16.empty;
sequences.ultra.shfl = containers.Map;
sequences.rb.orig = int16.empty;
sequences.rb.shfl = containers.Map;

for block_sz_id = 1:length(block_sizes)
    block_sz = int2str(block_sizes(block_sz_id));
    sequences.ultra.shfl(block_sz) = int16.empty;
    sequences.rb.shfl(block_sz) = int16.empty;
end

% Loading ultrametric sequences data
um_simusets = dir(um_root);
dirFlags = [um_simusets.isdir];
um_simusets = um_simusets(dirFlags);

for um_simuset_id = 1:length(um_simusets)
    um_simuset = um_simusets(um_simuset_id).name;
    matfiles_root = fullfile(um_root, um_simuset, 'matlab');
    matfiles = dir(matfiles_root);
    ndirFlags = ~[matfiles.isdir];
    matfiles = matfiles(ndirFlags);
    for matfile_id = 1:length(matfiles)
        matfilename = matfiles(matfile_id).name;
        new_seq = load(fullfile(um_root, um_simuset, 'matlab', matfilename));
        new_seq = new_seq.sequence;
        new_seq = new_seq(1,1:300000);
        for block_sz_id = 1:length(block_sizes)
            block_sz = int2str(block_sizes(block_sz_id));
            if(contains(matfilename, strcat('_shfl_',block_sz,'.mat')))
                sequences.ultra.shfl(block_sz) = [sequences.ultra.shfl(block_sz);new_seq];
                break;
            end
        end

        if(contains(matfilename, strcat('_orig.mat')))
            sequences.ultra.orig = [sequences.ultra.orig;new_seq];
        end
    end
end

% Loading random block sequences data
rb2_simusets = dir(rb2_root);
dirFlags = [rb2_simusets.isdir];
rb2_simusets = rb2_simusets(dirFlags);

for rb2_simuset_id = 1:length(rb2_simusets)
    rb2_simuset = rb2_simusets(rb2_simuset_id).name;
    matfiles_root = fullfile(rb2_root, rb2_simuset, 'matlab');
    matfiles = dir(matfiles_root);
    ndirFlags = ~[matfiles.isdir];
    matfiles = matfiles(ndirFlags);
    for matfile_id = 1:length(matfiles)
        matfilename = matfiles(matfile_id).name;
        new_seq = load(fullfile(rb2_root, rb2_simuset, 'matlab', matfilename));
        new_seq = new_seq.sequence;
        new_seq = new_seq(1,1:300000);
        for block_sz_id = 1:length(block_sizes)
            block_sz = int2str(block_sizes(block_sz_id));
            if(contains(matfilename, strcat('_shfl_',block_sz,'.mat')))
                sequences.rb.shfl(block_sz) = [sequences.rb.shfl(block_sz);new_seq];
                break;
            end
        end

        if(contains(matfilename, strcat('_orig.mat')))
            sequences.rb.orig = [sequences.rb.orig;new_seq];
        end
    end
end

% compute P_0 for ultrametric sequences

hlocs_stat_um=zeros(length(block_sizes)+1,maxh-1);

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
            hlocs_stat_um(1,:)=hlocs_stat_um(1,:)+hlocs./tree_l;
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
            hlocs_stat_um(1+block_sz_id,:)=hlocs_stat_um(1+block_sz_id,:)+hlocs./tree_l;
            end
        end
    end
end


% compute P_0 for random_blocks sequences

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

% Saving mat files
atc_um_filename = sprintf("atc_um_%s_%s_%s.mat", dataset_name, nnarchi, datetime('now','Format','yyyyMMMdd'));
atc_rb2_filename = sprintf("atc_rb2_%s_%s_%s.mat", dataset_name, nnarchi, datetime('now','Format','yyyyMMMdd'));
save(atc_um_filename, 'hlocs_stat_um');
save(atc_rb2_filename, 'hlocs_stat_rb');

% Plotting results
plot_autocorr(atc_um_filename, atc_rb2_filename, block_sizes);

end

