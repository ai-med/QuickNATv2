function visualize_segmentation(stats, imdb, net, opts)
	epoch = stats.epoch;
   

	plotErrors(opts, epoch, stats, false);
    drawConfusionMatrix(opts, stats);
	showImages(imdb, net, opts); 
    
    % print to disk
    if(opts.savePlots)
        print(opts.vis.hnd_loss, sprintf('%s/summary.pdf',opts.expDir), '-dpdf'); 
        print(opts.vis.hand_examples, sprintf('%s/examples.pdf',opts.expDir), '-dpdf'); 
        print(opts.vis.hand_conf, sprintf('%s/confm.pdf',opts.expDir), '-dpdf');
    end
end

function drawConfusionMatrix(opts, stats)
	change_current_figure(opts.vis.hand_conf); clf;
    
    confT = stats.train(end).confM;
     save('confT.mat','confT');
%     confJac = stats.train(end).confJac;
%     confJac = confJac;
    for l=1:size(confT, 1)
		confT(l, :) = confT(l, :) / (sum(confT(l, :))+eps);
%         confJac(l, :) = confJac(l, :) / (sum(confJac(l, :))+eps);
    end
    
    confV = stats.val(end).confM;
    for l=1:size(confV, 1)
		confV(l, :) = confV(l, :) / (sum(confV(l, :))+eps);
    end
   
%     save('confJac.mat','confJac');
    
    subplot(2,1,1);
	imagesc(confT);
    colormap(jet);
    caxis([0, 1])
	textstr=num2str(confT(:),'%0.2f');
	textstr=strtrim(cellstr(textstr));
	L = size(confT, 1);
	[x,y]=meshgrid(1:L);
	hstrg=text(x(:),y(:),textstr(:), 'FontWeight','bold', 'HorizontalAlignment','center','FontSize',8,'FontName','Times New Roman');
	midvalue=mean(get(gca,'Clim'));
	textColors=repmat([1, 1, 1], numel(confT), 1);
	set(hstrg,{'color'},num2cell(textColors,2));
	set(gca,'XTick',1:L,'XTickLabel',opts.classesNames,'YTick',1:L,'YTickLabel',opts.classesNames,'TickLength',[0,0],'FontSize',6,'FontName','Times New Roman', 'FontWeight','bold');
	colorbar;
	xlabel('Predicted class');
	ylabel('True class');
	title('Training');
    
    
    subplot(2,1,2);
	imagesc(confV);
    colormap(jet);
    caxis([0, 1])
	textstr=num2str(confV(:),'%0.2f');
	textstr=strtrim(cellstr(textstr));
	L = size(confV, 1);
	[x,y]=meshgrid(1:L);
	hstrg=text(x(:),y(:),textstr(:), 'FontWeight','bold', 'HorizontalAlignment','center','FontSize',8,'FontName','Times New Roman');
	midvalue=mean(get(gca,'Clim'));
	textColors=repmat([1, 1, 1], numel(confV), 1);
	set(hstrg,{'color'},num2cell(textColors,2));
	set(gca,'XTick',1:L,'XTickLabel',opts.classesNames,'YTick',1:L,'YTickLabel',opts.classesNames,'TickLength',[0,0],'FontSize',6,'FontName','Times New Roman', 'FontWeight','bold');
	colorbar;
	xlabel('Predicted class');
	ylabel('True class');
	title('Validation');
end


%--------------------------------------------------------------------------
function plotErrors(opts, epoch, stats, evaluateMode)
% -------------------------------------------------------------------------
    expDir = opts.expDir;
    opts = opts.vis;
    
 	change_current_figure(opts.hnd_loss); clf;
	sets = {'train', 'val'};

    % plot for the objective
    subplot(1,2,1);
	if ~evaluateMode
        semilogy(1:epoch, gather([stats.train.objective])./[stats.train.num_batches], '.-', 'linewidth', 2);
        hold on;
    end
    
	semilogy(1:epoch, gather([stats.val.objective])./[stats.val.num_batches], '.--');
	xlabel('training epoch'); ylabel('energy');
	grid on ;
	h=legend(sets) ;
	set(h,'color','none');
	title('objective') ;
   
    t_overalls = [stats.train.overall_global];
    t_perclasses = [stats.train.perclass_global];
    v_overalls = [stats.val.overall_global];
    v_perclasses = [stats.val.perclass_global];
    
	subplot(1,2,2) ; leg = {} ;
	if ~evaluateMode
	    plot(1:epoch, 100-[t_overalls', t_perclasses'], '.-', 'linewidth', 2) ;
	    hold on ;
	    leg = {'train. overall', 'train. perclass'} ;
	end

	plot(1:epoch, 100-[v_overalls',  v_perclasses'], '.--') ;
	leg{end+1} = 'val. overall';
    leg{end+1} = 'val. perclass' ;
	set(legend(leg{:}),'color','none') ;
	grid on ;
	xlabel('training epoch') ; ylabel('error') ;
	title('error') ;
 	drawnow ;
% 	if(exist(opts.modelFigPath, 'file'))
% 	delete(opts.modelFigPath);
% 	end
%	print(opts.hnd_loss, sprintf('%s/summary.pdf',expDir), '-dpdf') ;
end


function showImages(imdb, net, opts)
  	change_current_figure(opts.vis.hand_examples);
	colorMapEst = opts.colorMapEst; 
    colorMapGT = opts.colorMapGT;
    indices = opts.exampleIndices;
    
    % create inputs
    images = imdb.images.data(:,:,:, indices) ;
	labels = imdb.images.label(:, :, 1, indices) ;
	if numel(opts.gpus) >  0
  		images = gpuArray(images);
        net.move('gpu') ;
    end
	inputs = {'input', images, 'label', labels};
    
    % run net
    net.conserveMemory = true;
    net.eval({'input', images});
    net.conserveMemory = false;

	% obtain the CNN otuput
	scores = net.vars(net.getVarIndex('prob')).value;
	scores = squeeze(gather(scores)); 
    [outW, outL] = max(scores, [], 3);



	for i=1:length(indices)
		subplot(length(indices), 3, (3*i)-2);
		im = imdb.images.data(:, :, :, indices(i));
		min_im = min(im(:));
		max_im = max(im(:));
		im = (im - min_im) / (max_im - min_im + 1);
		imshow(im);
		title('RGB');
		subplot(length(indices), 3, (3*i)-1);
		imshow(squeeze(outL(:,:,:,i)), colorMapEst);
		freezeColors;
		title('Estimated labels');
		subplot(length(indices), 3, 3*i);
		imshow(squeeze(labels(:,:,:,i))+1, colorMapGT);
		title('Ground truth');
	end
	refreshdata;
	drawnow;
end


function change_current_figure(h)
	set(0,'CurrentFigure',h)
end

