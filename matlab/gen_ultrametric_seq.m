% Ultrametric transition process

L = 5
N = 2^L;
Steps = 10^4

figure(1)
clf

for T = [0.1:0.05:0.8]

TMat = zeros(N,N);
for l=L:-1:0
    for b=1:2^(L-l)        
        TMat((b-1)*2^l+1:b*2^l,(b-1)*2^l+1:b*2^l) = 2^(0-l/T);
    end
end
%TMat = TMat - eye(N);

X = cat(1,1,zeros(N-1,1));
XStore = zeros(1,Steps);
for s=1:Steps
    X = TMat*X/sum(TMat*X);
    XStore(1,s) = X(1);
end
X

figure(1)
loglog(XStore(1,:))
hold on

end

figure(1)
ylabel('Occupancy probability - Autocorrelation function')
hold off