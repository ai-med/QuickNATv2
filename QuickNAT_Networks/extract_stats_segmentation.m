function stats = extract_stats_segmentation(net, stats)
	sel = find(cellfun(@(x) isa(x,'dagnn.LossSemantic'), {net.layers.block})) ;

    obj = net.vars(net.layers(sel).outputIndexes(1)).value;
	predictions = net.vars(net.layers(sel).inputIndexes(1)).value;
	labels = net.vars(net.layers(sel).inputIndexes(2)).value;
	L = size(predictions, 3);

	[mm, preds] = max(predictions, [], 3);
	preds = squeeze(preds);

	p = gather(preds(:));
    labels = labels(:,:,1,:);
	l = gather(labels(:));

	confM = zeros(L, L);
	for r=1:L % gt
		for c=1:L % pred
			confM(r, c) = sum((l == r) & (p == c));
		end
    end
    
    % total number of elements ~= 0
	N = sum(confM(:));
	% overall (current minibatch)
	overall = 0; for r=1:L, overall = overall + confM(r,r); end; overall = 100*overall / N;
	% per-class (current minibatch)
	perclass = 0; for r=1:L, perclass = perclass + confM(r,r)/(sum(confM(r,:))+eps); end; perclass = 100*perclass / L;
    
    % confM is accumulated using all batches
    if(isfield(stats, 'confM'))
        stats.confM = stats.confM + confM;
    else
        stats.confM = confM;
    end
   
    % local overall and perclass
    stats.overall = overall;
    stats.perclass = perclass;
    
    % global overall and perclass (so-far)
    % total number of elements ~= 0
    confM = stats.confM;
	N = sum(confM(:));
	% overall (current minibatch)
	overall = 0; for r=1:L, overall = overall + confM(r,r); end; overall = 100*overall / N;
	% per-class (current minibatch)
	perclass = 0; for r=1:L, perclass = perclass + confM(r,r)/(sum(confM(r,:))+eps); end; perclass = 100*perclass / L;
 
    stats.overall_global = overall;
    stats.perclass_global = perclass;
    
    
    if(isfield(stats, 'objective'))
        stats.objective = obj + stats.objective;
    else
        stats.objective = obj;
    end
    
end
