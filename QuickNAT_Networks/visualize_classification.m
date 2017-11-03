function visualize_classification(opts)
	change_current_figure(opts.hand); clf;

	stats = opts.stats;
	epoch = opts.epoch;	

	plots = setdiff(cat(2, fieldnames(stats.train)', fieldnames(stats.val)'), {'num', 'time'}) ;
	for p = plots
		p = char(p) ;
		values = zeros(0, epoch) ;
		leg = {} ;
		for f = {'train', 'val'}
			f = char(f) ;
			if isfield(stats.(f), p)
				tmp = [stats.(f).(p)] ;
				values(end+1,:) = tmp(1,:)' ;
				leg{end+1} = f ;
			end
		end
	
		subplot(1,numel(plots),find(strcmp(p,plots))) ;
		plot(1:epoch, values','o-') ;
		xlabel('epoch') ;
		title(p) ;
		legend(leg{:}) ;
		grid on ;
	end
	drawnow ;

	% save stats
	print(1, opts.modelFigPath, '-dpdf') ;
end


function change_current_figure(h)
	set(0,'CurrentFigure',h)
end

