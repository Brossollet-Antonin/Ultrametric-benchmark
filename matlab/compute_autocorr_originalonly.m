function [ output_args ] = compute_autocorr_originalonly( dataset_name, nnarchi, um_battery, rb2_battery, maxh, max_seqlen )
%COMPUTE_AUTOCORR Summary of this function goes here
%   Detailed explanation goes here

project_root='~/ultrametric_benchmark/Ultrametric-benchmark';
data_root=fullfile(project_root,'Results',dataset_name,nnarchi);
um_root=fullfile(data_root,um_battery);
rb2_root=fullfile(data_root,rb2_battery);

% Getting parameter files
um_params = jsondecode(fileread(fullfile(um_root, 'parameters.json')));
rb2_params = jsondecode(fileread(fullfile(rb2_root, 'parameters.json')));

% Checking consistency
assert(um_params.TreeDepth==rb2_params.TreeDepth);
assert(um_params.TreeBranching==rb2_params.TreeBranching);

command=um_params.OriginalCommand;

% Reading block sizes, tree size, etc
tree_levels=um_params.TreeDepth;
branching=um_params.TreeBranching;
seq_length=um_params.SequenceLength;
tree_l=branching^tree_levels;

% Initializing objects
sequences.ultra.orig = int16.empty;
sequences.rb.orig = int16.empty;

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
        new_seq = new_seq(1,1:seq_length);
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
        new_seq = new_seq(1,1:seq_length);
        if(contains(matfilename, strcat('_orig.mat')))
            sequences.rb.orig = [sequences.rb.orig;new_seq];
        end
    end
end

% compute P_0 for ultrametric sequences

hlocs_stat_um=zeros(1,maxh);

disp('Ultrametric - Computing P0 for original sequences...')
for seq_id = 1:size(sequences.ultra.orig,1)
    lbl_seq = sequences.ultra.orig(seq_id,1:max_seqlen);
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
            edges=1:maxh+1;
            hlocs= histcounts(locsd(1:(nd-1)),edges); % compute the histogram of the lags
            hlocs_stat_um(1,:)=hlocs_stat_um(1,:)+hlocs./(size(sequences.ultra.orig,1));
        end
    end
end
% Normalizing with number of possible label matches
hlocs_stat_um(1,:) = hlocs_stat_um(1,:) ./ linspace(seq_length-1,seq_length-maxh,maxh);

% Rectifying ultrametric sequence
%hlocs_stat_um = [1 hlocs_stat_um];
%hlocs_stat_um(1,2:2:end) = sqrt(hlocs_stat_um(1,1:2:end-1).*hlocs_stat_um(1,3:2:end));

% compute P_0 for random_blocks sequences

hlocs_stat_rb=zeros(1,maxh);

disp('Random blocks - Computing P0 for original sequences...')
for seq_id = 1:size(sequences.rb.orig,1)
    lbl_seq = sequences.rb.orig(seq_id,1:max_seqlen);
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
            edges=1:maxh+1;
            hlocs= histcounts(locsd(1:(nd-1)),edges); % compute the histogram of the lags
            hlocs_stat_rb(1,:)=hlocs_stat_rb(1,:)+hlocs./size(sequences.rb.orig,1);
        end
    end
end
% Normalizing with number of possible label matches
hlocs_stat_rb(1,:) = hlocs_stat_rb(1,:) ./ linspace(seq_length-1,seq_length-maxh,maxh);
%hlocs_stat_rb = [1 hlocs_stat_rb];

% Saving mat files
atc_um_filename = sprintf("atc_um_%s_%s_%s.mat", dataset_name, nnarchi, datetime('now','Format','yyyyMMMdd'));
atc_rb2_filename = sprintf("atc_rb2_%s_%s_%s.mat", dataset_name, nnarchi, datetime('now','Format','yyyyMMMdd'));
save(atc_um_filename, 'hlocs_stat_um');
save(atc_rb2_filename, 'hlocs_stat_rb');

% Plotting results
% plot_autocorr(atc_um_filename, atc_rb2_filename, block_sizes);

end

