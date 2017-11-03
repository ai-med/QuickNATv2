function [net, info] = QuickNAT_Train(imdb, netF, inpt, varargin)

	% some common options
	trainer = @cnn_train_dag_seg;

	opts.train.extractStatsFn = @extract_stats_segmentation_Mod;
	opts.train.batchSize = 4;
	opts.train.numEpochs = 15;
	opts.train.continue = true ;
	opts.train.gpus = [2] ;
	opts.train.learningRate = [1e-1*ones(1, 5),  1e-2*ones(1, 5),  1e-3*ones(1, 5), 1e-4*ones(1,1)];
	opts.train.weightDecay = 1e-3;
	opts.train.momentum = 0.95;
	opts.train.expDir = inpt.expDir;
	opts.train.savePlots = false;
	opts.train.numSubBatches = 1;
	% getBatch options
	bopts.useGpu = numel(opts.train.gpus) >  0 ;

    
    opts.border = [8 8 8 8]; % tblr

    
    % augmenting data - Jitter and Fliplr
    augData = zeros(size(imdb.images.data) + [sum(opts.border(1:2)) ...
    sum(opts.border(3:4)) 0 0], 'like', imdb.images.data);
    augLabels = zeros(size(imdb.images.label) + [sum(opts.border(1:2)) ...
    sum(opts.border(3:4)) 0 0], 'like', imdb.images.label);
    augData(opts.border(1)+1:end-opts.border(2), ...
    opts.border(3)+1:end-opts.border(4), :, :) = imdb.images.data;
    % Mirroring Borders for augData
    augData(1:opts.border(1), ...
    opts.border(3)+1:end-opts.border(4), :, :) = imdb.images.data(opts.border(1):-1:1, ...
    :, :, :);
    augData(end-opts.border(2)+1:end, ...
    opts.border(3)+1:end-opts.border(4), :, :) = imdb.images.data(end:-1:end-opts.border(2)+1, ...
    :, :, :);
    augData(:, ...
    opts.border(3):-1:1, :, :) = augData(:, ...
    opts.border(3)+1:2*opts.border(3), :, :);
    augData(:, ...
    end-opts.border(4)+1:end, :, :) = augData(:, ...
    end-opts.border(4):-1:end-2*opts.border(4)+1, :, :);

    % Augmenting Labels
    augLabels(opts.border(1)+1:end-opts.border(2), ...
    opts.border(3)+1:end-opts.border(4), :, :) = imdb.images.label;
    % Mirroring Borders for augLabels
    augLabels(1:opts.border(1), ...
    opts.border(3)+1:end-opts.border(4), :, :) = imdb.images.label(opts.border(1):-1:1, ...
    :, :, :);
    augLabels(end-opts.border(2)+1:end, ...
    opts.border(3)+1:end-opts.border(4), :, :) = imdb.images.label(end:-1:end-opts.border(2)+1, ...
    :, :, :);
    augLabels(:, ...
    opts.border(3):-1:1, :, :) = augLabels(:, ...
    opts.border(3)+1:2*opts.border(3), :, :);
    augLabels(:, ...
    end-opts.border(4)+1:end, :, :) = augLabels(:, ...
    end-opts.border(4):-1:end-2*opts.border(4)+1, :, :);
    

    imdb.images.augData = augData;
    imdb.images.augLabels = augLabels;
    clear augData augLabels
    
    
	% organize data
	K = 2; % how many examples per domain	
	trainData = find(imdb.images.set == 1);
	valData = find(imdb.images.set == 3);
	
	% debuging code
	opts.train.exampleIndices = [trainData(randperm(numel(trainData), K)), valData(randperm(numel(valData), K))];

% 	opts.train.classesNames = {'sky', 'building', 'road', 'sidewalk', 'fence', 'vegetation', 'pole', 'car', 'sign', 'pedestrian', 'cyclist'};
	colorMap  = (1/255)*[		    
					    128 128 128
					    128 0 0
					    128 64 128
					    0 0 192
					    64 64 128
					    128 128 0
					    192 192 128
					    64 0 128
					    192 128 128
					    64 64 0
					    0 128 192
                        128 128 128
					    128 0 0
					    128 64 128
					    0 0 192
					    64 64 128
					    128 128 0
					    192 192 128
					    64 0 128
					    192 128 128
					    64 64 0
					    0 128 192
                        128 128 128
					    128 0 0
					    128 64 128
					    0 0 192
					    64 64 128
					    128 128 0
					    ];
	opts.train.colorMapGT = [0 0 0; colorMap];
	opts.train.colorMapEst = colorMap;

	% network definition
	net = dagnn.DagNN() ;
    % Dense Encoder 1
    net.addLayer('bn1_1', dagnn.BatchNorm('numChannels', 1), {'input'}, {'bn1_1'}, {'bn1_1f', 'bn1_1b', 'bn1_1m'});
	net.addLayer('relu1_1', dagnn.ReLU(), {'bn1_1'}, {'relu1_1'}, {});
	net.addLayer('conv1_1', dagnn.Conv('size', [5 5 1 64], 'hasBias', true, 'stride', [1, 1], 'pad', [2 2 2 2]), {'relu1_1'}, {'conv1_1'},  {'conv1_1f'  'conv1_1b'});
    net.addLayer('concat1_1', dagnn.Concat('dim',3), {'input','conv1_1'}, {'concat1_1'});
    net.addLayer('bn1_2', dagnn.BatchNorm('numChannels', 65), {'concat1_1'}, {'bn1_2'}, {'bn1_2f', 'bn1_2b', 'bn1_2m'});
    net.addLayer('relu1_2', dagnn.ReLU(), {'bn1_2'}, {'relu1_2'}, {});
    net.addLayer('conv1_2', dagnn.Conv('size', [5 5 65 64], 'hasBias', true, 'stride', [1, 1], 'pad', [2 2 2 2]), {'relu1_2'}, {'conv1_2'},  {'conv1_2f'  'conv1_2b'});
    net.addLayer('concat1_2', dagnn.Concat('dim',3), {'input','conv1_1', 'conv1_2'}, {'concat1_2'});
    net.addLayer('bn1_3', dagnn.BatchNorm('numChannels', 129), {'concat1_2'}, {'bn1_3'}, {'bn1_3f', 'bn1_3b', 'bn1_3m'});
    net.addLayer('relu1_3', dagnn.ReLU(), {'bn1_3'}, {'relu1_3'}, {});
    net.addLayer('conv1_3', dagnn.Conv('size', [1 1 129 64], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'relu1_3'}, {'conv1_3'},  {'conv1_3f'  'conv1_3b'});
	net.addLayer('pool1', dagnn.PoolingInd('method', 'max', 'poolSize', [2, 2], 'stride', [2, 2], 'pad', [0 0 0 0]), {'conv1_3'}, {'pool1', 'pool_indices_1', 'sizes_pre_pool_1', 'sizes_post_pool_1'}, {});

    % Dense Encoder 2
	net.addLayer('bn2_1', dagnn.BatchNorm('numChannels', 64), {'pool1'}, {'bn2_1'}, {'bn2_1f', 'bn2_1b', 'bn2_1m'});
	net.addLayer('relu2_1', dagnn.ReLU(), {'bn2_1'}, {'relu2_1'}, {});
	net.addLayer('conv2_1', dagnn.Conv('size', [5 5 64 64], 'hasBias', true, 'stride', [1, 1], 'pad', [2 2 2 2]), {'relu2_1'}, {'conv2_1'},  {'conv2_1f'  'conv2_1b'});
    net.addLayer('concat2_1', dagnn.Concat('dim',3), {'pool1','conv2_1'}, {'concat2_1'});
    net.addLayer('bn2_2', dagnn.BatchNorm('numChannels', 128), {'concat2_1'}, {'bn2_2'}, {'bn2_2f', 'bn2_2b', 'bn2_2m'});
    net.addLayer('relu2_2', dagnn.ReLU(), {'bn2_2'}, {'relu2_2'}, {});
    net.addLayer('conv2_2', dagnn.Conv('size', [5 5 128 64], 'hasBias', true, 'stride', [1, 1], 'pad', [2 2 2 2]), {'relu2_2'}, {'conv2_2'},  {'conv2_2f'  'conv2_2b'});
    net.addLayer('concat2_2', dagnn.Concat('dim',3), {'pool1','conv2_1', 'conv2_2'}, {'concat2_2'});
    net.addLayer('bn2_3', dagnn.BatchNorm('numChannels', 192), {'concat2_2'}, {'bn2_3'}, {'bn2_3f', 'bn2_3b', 'bn2_3m'});
    net.addLayer('relu2_3', dagnn.ReLU(), {'bn2_3'}, {'relu2_3'}, {});
    net.addLayer('conv2_3', dagnn.Conv('size', [1 1 192 64], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'relu2_3'}, {'conv2_3'},  {'conv2_3f'  'conv2_3b'});
	net.addLayer('pool2', dagnn.PoolingInd('method', 'max', 'poolSize', [2, 2], 'stride', [2, 2], 'pad', [0 0 0 0]), {'conv2_3'}, {'pool2', 'pool_indices_2', 'sizes_pre_pool_2', 'sizes_post_pool_2'}, {});

    % Dense Encoder 3
    net.addLayer('bn3_1', dagnn.BatchNorm('numChannels', 64), {'pool2'}, {'bn3_1'}, {'bn3_1f', 'bn3_1b', 'bn3_1m'});
	net.addLayer('relu3_1', dagnn.ReLU(), {'bn3_1'}, {'relu3_1'}, {});
	net.addLayer('conv3_1', dagnn.Conv('size', [5 5 64 64], 'hasBias', true, 'stride', [1, 1], 'pad', [2 2 2 2]), {'relu3_1'}, {'conv3_1'},  {'conv3_1f'  'conv3_1b'});
    net.addLayer('concat3_1', dagnn.Concat('dim',3), {'pool2','conv3_1'}, {'concat3_1'});
    net.addLayer('bn3_2', dagnn.BatchNorm('numChannels', 128), {'concat3_1'}, {'bn3_2'}, {'bn3_2f', 'bn3_2b', 'bn3_2m'});
    net.addLayer('relu3_2', dagnn.ReLU(), {'bn3_2'}, {'relu3_2'}, {});
    net.addLayer('conv3_2', dagnn.Conv('size', [5 5 128 64], 'hasBias', true, 'stride', [1, 1], 'pad', [2 2 2 2]), {'relu3_2'}, {'conv3_2'},  {'conv3_2f'  'conv3_2b'});
    net.addLayer('concat3_2', dagnn.Concat('dim',3), {'pool2','conv3_1', 'conv3_2'}, {'concat3_2'});
    net.addLayer('bn3_3', dagnn.BatchNorm('numChannels', 192), {'concat3_2'}, {'bn3_3'}, {'bn3_3f', 'bn3_3b', 'bn3_3m'});
    net.addLayer('relu3_3', dagnn.ReLU(), {'bn3_3'}, {'relu3_3'}, {});
    net.addLayer('conv3_3', dagnn.Conv('size', [1 1 192 64], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'relu3_3'}, {'conv3_3'},  {'conv3_3f'  'conv3_3b'});
	net.addLayer('pool3', dagnn.PoolingInd('method', 'max', 'poolSize', [2, 2], 'stride', [2, 2], 'pad', [0 0 0 0]), {'conv3_3'}, {'pool3', 'pool_indices_3', 'sizes_pre_pool_3', 'sizes_post_pool_3'}, {});

    % Dense Encoder 4
    net.addLayer('bn4_1', dagnn.BatchNorm('numChannels', 64), {'pool3'}, {'bn4_1'}, {'bn4_1f', 'bn4_1b', 'bn4_1m'});
	net.addLayer('relu4_1', dagnn.ReLU(), {'bn4_1'}, {'relu4_1'}, {});
	net.addLayer('conv4_1', dagnn.Conv('size', [5 5 64 64], 'hasBias', true, 'stride', [1, 1], 'pad', [2 2 2 2]), {'relu4_1'}, {'conv4_1'},  {'conv4_1f'  'conv4_1b'});
    net.addLayer('concat4_1', dagnn.Concat('dim',3), {'pool3','conv4_1'}, {'concat4_1'});
    net.addLayer('bn4_2', dagnn.BatchNorm('numChannels', 128), {'concat4_1'}, {'bn4_2'}, {'bn4_2f', 'bn4_2b', 'bn4_2m'});
    net.addLayer('relu4_2', dagnn.ReLU(), {'bn4_2'}, {'relu4_2'}, {});
    net.addLayer('conv4_2', dagnn.Conv('size', [5 5 128 64], 'hasBias', true, 'stride', [1, 1], 'pad', [2 2 2 2]), {'relu4_2'}, {'conv4_2'},  {'conv4_2f'  'conv4_2b'});
    net.addLayer('concat4_2', dagnn.Concat('dim',3), {'pool3','conv4_1', 'conv4_2'}, {'concat4_2'});
    net.addLayer('bn4_3', dagnn.BatchNorm('numChannels', 192), {'concat4_2'}, {'bn4_3'}, {'bn4_3f', 'bn4_3b', 'bn4_3m'});
    net.addLayer('relu4_3', dagnn.ReLU(), {'bn4_3'}, {'relu4_3'}, {});
    net.addLayer('conv4_3', dagnn.Conv('size', [1 1 192 64], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'relu4_3'}, {'conv4_3'},  {'conv4_3f'  'conv4_3b'});
	net.addLayer('pool4', dagnn.PoolingInd('method', 'max', 'poolSize', [2, 2], 'stride', [2, 2], 'pad', [0 0 0 0]), {'conv4_3'}, {'pool4', 'pool_indices_4', 'sizes_pre_pool_4', 'sizes_post_pool_4'}, {});

    
    % BottleNeck Layers
    net.addLayer('conv5', dagnn.Conv('size', [5 5 64 64], 'hasBias', true, 'stride', [1, 1], 'pad', [2 2 2 2]), {'pool4'}, {'conv5'},  {'conv5f'  'conv5b'});
	net.addLayer('bn5', dagnn.BatchNorm('numChannels', 64), {'conv5'}, {'bn5'}, {'bn5f', 'bn5b', 'bn5m'});
    
    % Dense Decoder 4
    net.addLayer('unpool4x', dagnn.Unpooling(), {'bn5', 'pool_indices_4', 'sizes_pre_pool_4', 'sizes_post_pool_4'}, {'unpool4x'}, {});
    net.addLayer('concat4x', dagnn.Concat('dim',3), {'unpool4x','conv4_3'}, {'concat4x'});
    net.addLayer('bn4_1x', dagnn.BatchNorm('numChannels', 128), {'concat4x'}, {'bn4_1x'}, {'bn4_1fx', 'bn4_1bx', 'bn4_1mx'});
	net.addLayer('relu4_1x', dagnn.ReLU(), {'bn4_1x'}, {'relu4_1x'}, {});
	net.addLayer('deconv4_1x', dagnn.Conv('size', [5 5 128 64], 'hasBias', true, 'stride', [1,1], 'pad', [2 2 2 2]), {'relu4_1x'}, {'deconv4_1x'},  {'deconv4_1fx'  'deconv4_1bx'});
    net.addLayer('concat4_1x', dagnn.Concat('dim',3), {'concat4x','deconv4_1x'}, {'concat4_1x'});
    net.addLayer('bn4_2x', dagnn.BatchNorm('numChannels', 192), {'concat4_1x'}, {'bn4_2x'}, {'bn4_2fx', 'bn4_2bx', 'bn4_2mx'});
	net.addLayer('relu4_2x', dagnn.ReLU(), {'bn4_2x'}, {'relu4_2x'}, {});
    net.addLayer('deconv4_2x', dagnn.Conv('size', [5 5 192 64], 'hasBias', true, 'stride', [1,1], 'pad', [2 2 2 2]), {'relu4_2x'}, {'deconv4_2x'},  {'deconv4_2fx'  'deconv4_2bx'});
	net.addLayer('concat4_2x', dagnn.Concat('dim',3), {'concat4x','deconv4_1x','deconv4_2x'}, {'concat4_2x'});
    net.addLayer('bn4_3x', dagnn.BatchNorm('numChannels', 256), {'concat4_2x'}, {'bn4_3x'}, {'bn4_3fx', 'bn4_3bx', 'bn4_3mx'});
	net.addLayer('relu4_3x', dagnn.ReLU(), {'bn4_3x'}, {'relu4_3x'}, {});
    net.addLayer('deconv4_3x', dagnn.Conv('size', [1 1 256 64], 'hasBias', true, 'stride', [1,1], 'pad', [0 0 0 0]), {'relu4_3x'}, {'deconv4_3x'},  {'deconv4_3fx'  'deconv4_3bx'});
	
    
    % Dense Decoder 3
    net.addLayer('unpool3x', dagnn.Unpooling(), {'deconv4_3x', 'pool_indices_3', 'sizes_pre_pool_3', 'sizes_post_pool_3'}, {'unpool3x'}, {});
    net.addLayer('concat3x', dagnn.Concat('dim',3), {'unpool3x','conv3_3'}, {'concat3x'});
    net.addLayer('bn3_1x', dagnn.BatchNorm('numChannels', 128), {'concat3x'}, {'bn3_1x'}, {'bn3_1fx', 'bn3_1bx', 'bn3_1mx'});
	net.addLayer('relu3_1x', dagnn.ReLU(), {'bn3_1x'}, {'relu3_1x'}, {});
	net.addLayer('deconv3_1x', dagnn.Conv('size', [5 5 128 64], 'hasBias', true, 'stride', [1,1], 'pad', [2 2 2 2]), {'relu3_1x'}, {'deconv3_1x'},  {'deconv3_1fx'  'deconv3_1bx'});
    net.addLayer('concat3_1x', dagnn.Concat('dim',3), {'concat3x','deconv3_1x'}, {'concat3_1x'});
    net.addLayer('bn3_2x', dagnn.BatchNorm('numChannels', 192), {'concat3_1x'}, {'bn3_2x'}, {'bn3_2fx', 'bn3_2bx', 'bn3_2mx'});
	net.addLayer('relu3_2x', dagnn.ReLU(), {'bn3_2x'}, {'relu3_2x'}, {});
    net.addLayer('deconv3_2x', dagnn.Conv('size', [5 5 192 64], 'hasBias', true, 'stride', [1,1], 'pad', [2 2 2 2]), {'relu3_2x'}, {'deconv3_2x'},  {'deconv3_2fx'  'deconv3_2bx'});
	net.addLayer('concat3_2x', dagnn.Concat('dim',3), {'concat3x','deconv3_1x','deconv3_2x'}, {'concat3_2x'});
    net.addLayer('bn3_3x', dagnn.BatchNorm('numChannels', 256), {'concat3_2x'}, {'bn3_3x'}, {'bn3_3fx', 'bn3_3bx', 'bn3_3mx'});
	net.addLayer('relu3_3x', dagnn.ReLU(), {'bn3_3x'}, {'relu3_3x'}, {});
    net.addLayer('deconv3_3x', dagnn.Conv('size', [1 1 256 64], 'hasBias', true, 'stride', [1,1], 'pad', [0 0 0 0]), {'relu3_3x'}, {'deconv3_3x'},  {'deconv3_3fx'  'deconv3_3bx'});
	 
    % Dense Decoder 2
    net.addLayer('unpool2x', dagnn.Unpooling(), {'deconv3_3x', 'pool_indices_2', 'sizes_pre_pool_2', 'sizes_post_pool_2'}, {'unpool2x'}, {});
    net.addLayer('concat2x', dagnn.Concat('dim',3), {'unpool2x','conv2_3'}, {'concat2x'});
    net.addLayer('bn2_1x', dagnn.BatchNorm('numChannels', 128), {'concat2x'}, {'bn2_1x'}, {'bn2_1fx', 'bn2_1bx', 'bn2_1mx'});
	net.addLayer('relu2_1x', dagnn.ReLU(), {'bn2_1x'}, {'relu2_1x'}, {});
	net.addLayer('deconv2_1x', dagnn.Conv('size', [5 5 128 64], 'hasBias', true, 'stride', [1,1], 'pad', [2 2 2 2]), {'relu2_1x'}, {'deconv2_1x'},  {'deconv2_1fx'  'deconv2_1bx'});
    net.addLayer('concat2_1x', dagnn.Concat('dim',3), {'concat2x','deconv2_1x'}, {'concat2_1x'});
    net.addLayer('bn2_2x', dagnn.BatchNorm('numChannels', 192), {'concat2_1x'}, {'bn2_2x'}, {'bn2_2fx', 'bn2_2bx', 'bn2_2mx'});
	net.addLayer('relu2_2x', dagnn.ReLU(), {'bn2_2x'}, {'relu2_2x'}, {});
    net.addLayer('deconv2_2x', dagnn.Conv('size', [5 5 192 64], 'hasBias', true, 'stride', [1,1], 'pad', [2 2 2 2]), {'relu2_2x'}, {'deconv2_2x'},  {'deconv2_2fx'  'deconv2_2bx'});
	net.addLayer('concat2_2x', dagnn.Concat('dim',3), {'concat2x','deconv2_1x','deconv2_2x'}, {'concat2_2x'});
    net.addLayer('bn2_3x', dagnn.BatchNorm('numChannels', 256), {'concat2_2x'}, {'bn2_3x'}, {'bn2_3fx', 'bn2_3bx', 'bn2_3mx'});
	net.addLayer('relu2_3x', dagnn.ReLU(), {'bn2_3x'}, {'relu2_3x'}, {});
    net.addLayer('deconv2_3x', dagnn.Conv('size', [1 1 256 64], 'hasBias', true, 'stride', [1,1], 'pad', [0 0 0 0]), {'relu2_3x'}, {'deconv2_3x'},  {'deconv2_3fx'  'deconv2_3bx'});
    
    % Dense Decoder 1
    net.addLayer('unpool1x', dagnn.Unpooling(), {'deconv2_3x', 'pool_indices_1', 'sizes_pre_pool_1', 'sizes_post_pool_1'}, {'unpool1x'}, {});
    net.addLayer('concat1x', dagnn.Concat('dim',3), {'unpool1x','conv1_3'}, {'concat1x'});
    net.addLayer('bn1_1x', dagnn.BatchNorm('numChannels', 128), {'concat1x'}, {'bn1_1x'}, {'bn1_1fx', 'bn1_1bx', 'bn1_1mx'});
	net.addLayer('relu1_1x', dagnn.ReLU(), {'bn1_1x'}, {'relu1_1x'}, {});
	net.addLayer('deconv1_1x', dagnn.Conv('size', [5 5 128 64], 'hasBias', true, 'stride', [1,1], 'pad', [2 2 2 2]), {'relu1_1x'}, {'deconv1_1x'},  {'deconv1_1fx'  'deconv1_1bx'});
    net.addLayer('concat1_1x', dagnn.Concat('dim',3), {'concat1x','deconv1_1x'}, {'concat1_1x'});
    net.addLayer('bn1_2x', dagnn.BatchNorm('numChannels', 192), {'concat1_1x'}, {'bn1_2x'}, {'bn1_2fx', 'bn1_2bx', 'bn1_2mx'});
	net.addLayer('relu1_2x', dagnn.ReLU(), {'bn1_2x'}, {'relu1_2x'}, {});
    net.addLayer('deconv1_2x', dagnn.Conv('size', [5 5 192 64], 'hasBias', true, 'stride', [1,1], 'pad', [2 2 2 2]), {'relu1_2x'}, {'deconv1_2x'},  {'deconv1_2fx'  'deconv1_2bx'});
	net.addLayer('concat1_2x', dagnn.Concat('dim',3), {'concat1x','deconv1_1x','deconv1_2x'}, {'concat1_2x'});
    net.addLayer('bn1_3x', dagnn.BatchNorm('numChannels', 256), {'concat1_2x'}, {'bn1_3x'}, {'bn1_3fx', 'bn1_3bx', 'bn1_3mx'});
	net.addLayer('relu1_3x', dagnn.ReLU(), {'bn1_3x'}, {'relu1_3x'}, {});
    net.addLayer('deconv1_3x', dagnn.Conv('size', [1 1 256 64], 'hasBias', true, 'stride', [1,1], 'pad', [0 0 0 0]), {'relu1_3x'}, {'deconv1_3x'},  {'deconv1_3fx'  'deconv1_3bx'});


    % Classifier and Losses
	net.addLayer('classifier', dagnn.Conv('size', [1 1 64 16], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'deconv1_3x'}, {'classifier'},  {'classf'  'classb'});
	net.addLayer('prob', dagnn.SoftMax(), {'classifier'}, {'prob'}, {});
  	net.addLayer('objective1', dagnn.LossSemantic('weights', 1), {'prob','label'}, 'objective1');
    net.addLayer('objective2', dagnn.Loss_TVReg('weights', 0), {'prob','label'}, 'objective2');
    net.addLayer('objective3', dagnn.LossDice('weights', 1), {'prob','label'}, 'objective3');
	% -- end of the network

	% do the training!
	initNet(net, netF);
	net.conserveMemory = false;

	info = trainer(net, imdb, @(i,b) getBatch(bopts,i,b), opts.train, 'train', trainData, 'val', valData) ;
end


% function on charge of creating a batch of images + labels
function inputs = getBatch(opts, imdb, batch)

if imdb.images.set(batch(1))==1,  % training
  sz0 = size(imdb.images.augData);
  sz = size(imdb.images.data);
  loc = [randi(sz0(1)-sz(1)+1) randi(sz0(2)-sz(2)+1)];
  images = imdb.images.augData(loc(1):loc(1)+sz(1)-1, ...
    loc(2):loc(2)+sz(2)-1, :, batch); 
  labels = imdb.images.augLabels(loc(1):loc(1)+sz(1)-1, ...
    loc(2):loc(2)+sz(2)-1, :, batch); 
    
%     Id = imdb.images.label(:,:,1,:)==13;
%     imdb.images.label(:,:,2,:) = single(imdb.images.label(:,:,2,:) + 5.*Id);

else                              % validating / testing
  images = imdb.images.data(:,:,:,batch); 
  labels = imdb.images.label(:,:,:,batch); 
end

if opts.useGpu > 0
    images = gpuArray(images);
    labels = gpuArray(labels); 
end
inputs = {'input', images, 'label', labels} ;
end

function initNet(net, netF)
	net.initParams();

    % He Initialization for New Layers
    for k=1:length(net.layers)
        % is a convolution layer?
        if(strcmp(class(net.layers(k).block), 'dagnn.Conv'))
            f_ind = net.layers(k).paramIndexes(1);
            b_ind = net.layers(k).paramIndexes(2);
            
            [h,w,in,out] = size(net.params(f_ind).value);
            He_gain = 0.7*sqrt(2/(size(net.params(f_ind).value,1)*size(net.params(f_ind).value,2)*size(net.params(f_ind).value,3))); % sqrt(2/fan_in)
            net.params(f_ind).value = He_gain*randn(size(net.params(f_ind).value), 'single');
            net.params(f_ind).learningRate = 1;
            net.params(f_ind).weightDecay = 1;
            
            net.params(b_ind).value = zeros(size(net.params(b_ind).value), 'single');
            net.params(b_ind).learningRate = 0.5;
            net.params(b_ind).weightDecay = 1;
        end
    end
    
end

