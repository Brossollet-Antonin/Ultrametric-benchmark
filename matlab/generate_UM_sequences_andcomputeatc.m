clear all;

T=0.4;
energy_step=1.0;
beta=energy_step/T;

tree_levels=5;
n_leaves=2^tree_levels; % number of leaves
maxh=50000;
seq_length = 1500000;

blocklenghts=[10, 100, 1000, 10000];
nbl=size(blocklenghts,2);

hlocs_stat_ultra=zeros(1+nbl,maxh-1);

% -------------------------- %
% Building transition matrix %
% -------------------------- %
% NOTE: Transition matrix here is NO-SELF: diagonal elements are zero

markovm=zeros(n_leaves,n_leaves);

md=tree_levels;
markovm=fillmarkov(markovm,1,1,n_leaves,md);

markovme=exp(-beta.*markovm);

for i=1:n_leaves
    tot=sum(markovme(i,:))-1;
    markovme(i,:)=markovme(i,:)./tot;
    markovme(i,i)=0;
end

% ---------------------------------- %
% Generate n_sequences Markov chains %
% ---------------------------------- %

transition_probabilities = markovme;
starting_value = 1;
chain = zeros(1,seq_length);

chain(1)=starting_value;
for i=2:seq_length
    this_step_distribution = transition_probabilities(chain(i-1),:);
    cumulative_distribution = cumsum(this_step_distribution);
    r = rand();
    chain(i) = find(cumulative_distribution>r,1);
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
    hlocs_stat_ultra(1,:)=hlocs_stat_ultra(1,:)+hlocs./n_leaves;
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
        hlocs_stat_ultra(1+block_size_id,:)=hlocs_stat_ultra(1+block_size_id,:)+hlocs./n_leaves;
    end
end

% Saving mat files
atc_um_filename = sprintf("scratch_atc_um_%d_%.2f_%s.mat", tree_levels, T, datetime('now','Format','yyyyMMMdd'));
save(atc_um_filename, 'hlocs_stat_ultra');

%-------------------%
% Support functions %
%-------------------%

function mo=fillmarkov(m,i1,j1,l,md)
    l2=l/2;
    for i=i1:(i1+l2-1)
        for j=j1:(j1+l2-1)
            m(i,j+l2)=md;
            m(j+l2,i)=md;
        end
    end
    if(md>1)

        m=fillmarkov(m,1,1,l2,md-1);
        % m1=m;
        for jjn=1:l2
            for kkk=1:l2
                %jjn
                %kkk
                m(kkk+l2,jjn+l2)=m(kkk,jjn);

            end
        end
    end
    %m
    mo=m;
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