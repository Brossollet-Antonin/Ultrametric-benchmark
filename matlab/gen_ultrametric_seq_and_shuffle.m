% This method generates the ultrametric sequence, and then
%  1) analyzes the return time histogram without any shuffling
%  2) performs a set of shuffling (defined by a set of shuffling
% lengths) and completes the same return time histogram analysis for each shuffling length

clear all;
tree_levels=10; % 12
maxh=10000;
beta=2.; % .2
chain_length = 1000000; %10000000

tree_l=2^tree_levels; % number of leaves

markovm=zeros(tree_l,tree_l);

% generate the Markov transition matrix
md=tree_levels;
markovm=fillmarkov(markovm,1,1,tree_l,md);



markovme=exp(-beta.*markovm);

for i=1:tree_l
    tot=sum(markovme(i,:))-1;
    markovme(i,:)=markovme(i,:)./tot;
    markovme(i,i)=0;
end

% generate the markov chain
transition_probabilities = markovme; starting_value = 1;
chain = zeros(1,chain_length);
chain(1)=starting_value;
for i=2:chain_length
    if(mod(i,chain_length/100)==0)
        i/(chain_length/100)
    end
    this_step_distribution = transition_probabilities(chain(i-1),:);
    cumulative_distribution = cumsum(this_step_distribution);
    r = rand();
    chain(i) = find(cumulative_distribution>r,1);
end
%  provides chain = 1 2 1 2 1 2 1 2 1 1 2 1 2 1 2....

% compute P_0 for no shuffle
% This computes the distribution of return times accross all leaves
% (accumulated accross leaves)

hlocs_stat=zeros(1,maxh-1);
for i=1:tree_l
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
   
    hlocs_stat=hlocs_stat+hlocs./tree_l;
end


figure(1);
clf;
hold on;
plot(hlocs_stat(2:end-1)./hlocs_stat(2),'r-');
set(gca,'xscale','log');
set(gca,'yscale','log'); 
title('Avg return time probability - Original sequence')
xlabel('Iteration')
ylabel('Normalized return time probability')

% shuffled data
blocklengths=[1,10, 100, 1000];
nbl=size(blocklengths,2);

for bl=1:nbl
    blength=blocklengths(bl);
    
    chains=shuffleblocks(chain,blength);
    
    hlocs_stats=zeros(1,maxh-1);
    for i=1:tree_l
        locs=find(chains==i); %find all occurrences of i
        
        nd=1;
        for j=1:size(locs,2)
            for k=(j+1):size(locs,2)
                locsd(nd)=locs(k)-locs(j);
                nd=nd+1;
            end
        end

        edges=1:maxh;
        hlocs= histcounts(locsd,edges,'Normalization','probability'); % compute the histogram of the lags
        
        hlocs_stats=hlocs_stats+hlocs./tree_l;
    end
    
    h=plot(hlocs_stats(2:end-1)./hlocs_stats(2),'b-');
    set(h,'color',[0 0 1-bl*.1]);
    title_str = sprintf('Avg return time probability - %d shuffle block length', blength);
    title(title_str);
    xlabel('Iteration');
    ylabel('Normalized return time probability');
    legend('0','1','10','100','1000');
    
end

hold off
%set(gca,'xscale','log');
%set(gca,'yscale','log');



% -----------------------------------------------------------

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
% I would have just chosen a permutation, applied it to the block ids and then re-generated the whole sequence from there...

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






