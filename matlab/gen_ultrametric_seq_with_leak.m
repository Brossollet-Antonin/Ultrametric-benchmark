% This method generates the ultrametric sequence, but adds a diffusion term in the Markov process
%
%

tree_levels=10; % 12
maxh=1000;
beta=2.; % .2
chain_length = 1000000; %10000000

tree_l=2^tree_levels; % number of leaves
%clist=[0,.1,.01,.001];
markovm=zeros(tree_l,tree_l);

% generate the Markov transition matrix
md=tree_levels;
markovm=fillmarkov(markovm,1,1,tree_l,md);
ci=1;
for c=[0,.1,.01,.001]
    
    markovme=exp(-beta.*markovm)+c/tree_l;

    for i=1:tree_l
        tot=sum(markovme(i,:))-markovme(i,i);
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
        % At this stage we have the distribution of the 'return' stopping time
        % Autocorrelation is given by the series of product correlations of
        % this distribution
        
        hlocs_stat=hlocs_stat+hlocs./tree_l;
        % Averaging over all leaves
    end

    figure(2);
    clf;
    hold on;
    %plot(hlocs_stat(2:end-1)./hlocs_stat(2),'r-');
    h=plot(hlocs_stat(2:end-1),'r-');
    set(h,'color',[1-ci*.2,0,0],'linewidth',1.5);
    title_str = 'Avg return time probability';
    title(title_str);
    set(gca,'xscale','log');
    set(gca,'yscale','log');
    ci=ci+1;

end


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







