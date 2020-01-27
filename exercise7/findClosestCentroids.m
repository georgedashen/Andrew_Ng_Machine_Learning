function idx = findClosestCentroids(X, initial_centroids)

m = size(X,1);
idx = ones(m,1);
K = size(initial_centroids,1);
for i=1:m
    dist = zeros(K,1);
    for j=1:K
        dist(j) = sum((X(i,:)-initial_centroids(j,:)).^2);
    end
    [val,idx(i)] = min(dist);
end