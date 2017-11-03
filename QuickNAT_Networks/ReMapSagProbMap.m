function [ModProbMap] = ReMapSagProbMap(ProbMap)

ReMapInd = [1, 2, 3, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 4, 5, 6, 7, 8, 9, 10, 14, 15, 16 ];

sz = size(ProbMap);
ModProbMap = zeros(sz(1),sz(2),28,sz(4), 'single');

for i = 1:28
    ModProbMap(:,:,i,:) = ProbMap(:,:,ReMapInd(i),:);
end
