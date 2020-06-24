clear all;

sequential_blocksz = 1000;

tree_levels=5;
n_leaves=2^tree_levels; % number of leaves
maxh=50000;
seq_length = 1500000;

blocklenghts=[10, 100, 1000, 10000];
nbl=size(blocklenghts,2);

hlocs_stat_rb=zeros(1+nbl,maxh-1);

% -------------------------- %
% Building transition matrix %
% -------------------------- %
% NOTE: Transition matrix here is NO-SELF: diagonal elements are zero

markovm=getmarkov_rb(n_leaves);

% ---------------------------------- %
% Generate n_sequences Markov chains %
% ---------------------------------- %

transition_probabilities = markovm;
starting_value = 1;
chain = zeros(1,seq_length);

chain(1)=starting_value;
for i=2:seq_length
    if(rem(i,sequential_blocksz)==0)
        chain(i) = randi([1 n_leaves],1,1);
    else
        this_step_distribution = transition_probabilities(chain(i-1),:);
        cumulative_distribution = cumsum(this_step_distribution);
        r = rand();
        chain(i) = find(cumulative_distribution>r,1);
    end
end
%  provides chain = 1 2 1 2 1 2 1 2 1 1 2 1 2 1 2....

% --------------------------------------------------- %
% Compute P_0 for the original sequences (no shuffle) %
% --------------------------------------------------- %
fprintf('Computing P_0, no shuffle...\n')

for i=1:n_leaves
    locs=find(chain==i); %find all occurrences of i

    nd=1;
    for j=1:size(locs,2)
        for k=(j+1):size(locs,2)
            locsd(nd)=locs(k)-locs(j);
            nd=nd+1;
        end
    end

    edges=1:maxh;
    hlocs= histcounts(locsd,edges,'Normalization','probability'); % compute the histogram of the lags
    clear('locsd');
    hlocs_stat_rb(1,:)=hlocs_stat_rb(1,:)+hlocs./n_leaves;
end

% ---------------------------------------%
% Compute P_0 for the shuffled sequences %
% ---------------------------------------%
for block_size_id=1:nbl
    block_sz=blocklenghts(block_size_id);
    fprintf('Computing P_0, block size %d...\n', block_sz)

    % Shuffling original sequence with blocks of size block_sz...
    chain_shfl=shuffleblocks(chain,block_sz);

    for i=1:n_leaves
        locs=find(chain_shfl==i); %find all occurrences of i

        nd=1;
        for j=1:size(locs,2)
            for k=(j+1):size(locs,2)
                locsd(nd)=locs(k)-locs(j);
                nd=nd+1;
            end
        end

        edges=1:maxh;
        hlocs= histcounts(locsd,edges,'Normalization','probability'); % compute the histogram of the lags
        clear('locsd');
        hlocs_stat_rb(1+block_size_id,:)=hlocs_stat_rb(1+block_size_id,:)+hlocs./n_leaves;
    end
end

% Saving mat files
atc_rb_filename = sprintf("scratch_atc_rb_%d_%d_%s.mat", tree_levels, sequential_blocksz, datetime('now','Format','yyyyMMMdd'));
save(atc_rb_filename, 'hlocs_stat_rb');

%-------------------%
% Support functions %
%-------------------%

function markovm=getmarkov_rb(n_leaves)
    markovm = zeros(n_leaves,n_leaves);
    n_couples=n_leaves/2;
    for c_id=1:n_couples
        markovm(2*c_id - 1,2*c_id)=1;
        markovm(2*c_id,2*c_id - 1)=1;
    end
end


% ---------------------------------------------------------

function chains=shuffleblocks(chain,blockl)

    if blockl==1
        chains=chain(randperm(length(chain)));
        return;
    end

    nb=floor(size(chain,2)/blockl); % number of blocks
    ns=nb*10; % number of shuffles


    for is=1:ns
        fi1=randi([1 (nb-1)],1)*blockl;
        fi2=randi([1 (nb-1)],1)*blockl;

        while(fi2==fi1)
            fi2=randi([1 (nb-1)],1)*blockl;
        end
        chain_buf=chain(fi1:(fi1+blockl));
        chain(fi1:(fi1+blockl))=chain(fi2:(fi2+blockl));
        chain(fi2:(fi2+blockl))=chain_buf;
    end

    chains=chain;

end