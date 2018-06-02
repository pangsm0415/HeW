clear
load('retrievalSfM120k-gem-vgg.mat');
net = dagnn.DagNN.loadobj(net);
net.vars(31).precious = true;
%% gpu
use_gpu = [];  % use GPUs (array of GPUIDs), if empty use CPU
numGpus = numel(use_gpu);
if numGpus
    fprintf('>> Prepring GPU(s)...\n'); 
end

if numGpus >= 1
	if numGpus == 1
		gpuinfo = gpuDevice(use_gpu);
		net.move('gpu');
		fprintf('>>>> Running on GPU %s with Index %d\n', gpuinfo.Name, gpuinfo.Index);  
	else
		spmd
			gpuinfo = gpuDevice(use_gpu(labindex));
			fprintf('>>>> Running on GPU %s with Index %d\n', gpuinfo.Name, gpuinfo.Index);  
		end
	end
end

use_gpu = strcmp(net.device, 'gpu');
if use_gpu
    gpuarrayfun = @(x) gpuArray(x);
	gatherfun = @(x) gather(x);
else
	gpuarrayfun = @(x) x; % do not convert to gpuArray
	gatherfun = @(x) x; % do not gather
end
addpath('E:\holidays_up\')
D = dir('E:\holidays_up\');
Holidays_cnn_up = cell(1, 1491);
minsize = 67;
net.mode = 'test';
tic
for i = 1:1491
    Img_name = D(i+2).name;
    im = imresizemaxd(imread(Img_name),1024,0);
    im = single(im) - mean(net.meta.normalization.averageImage(:));
    if min(size(im, 1), size(im, 2)) < minsize
        im = pad2minsize(im, minsize, 0);
    end
    net.eval({'input', gpuarrayfun(reshape(im, [size(im), 1]))});
    Holidays_cnn_up{i} = net.vars(31).value;
end
toc    
save('Holidays_cnn_up','Holidays_cnn_up','-v7.3')
