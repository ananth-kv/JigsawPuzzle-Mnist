require 'torch';
require 'nn';
require 'cunn';
require 'optim';
require 'image';
require 'dataset';
require 'model';
require 'cutorch'

-----------------------------------------------------------------------------
--------------------- parse command line options ----------------------------
-----------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text("Options")
--data params
cmd:option("-train_data","/home/ananth/siamese_data/train.h5", "Training Data")
cmd:option("-val_data","/home/ananth/siamese_data/val.h5", "Validation Data")
cmd:option("-shuffle","data/shuffle.dat","Shuffle Data")
--
cmd:option("-max_epochs", 5000, "maximum epochs")
cmd:option("-batch_size", 200, "batch size")
cmd:option("-learning_rate", 0.001, "learning_rate")
cmd:option("-momentum", 0.9, "momentum")
cmd:option("-snapshot_dir", "/home/ananth/snapshot/", "snapshot directory")
cmd:option("-snapshot_epoch", 5, "snapshot after how many iterations?")
cmd:option("-gpu", true, "use gpu")
cmd:option("-weights", "", "pretrained model to begin training from")
cmd:option("-log","train_new.log" ,"output log file")
cmd:option("-GPU",1 ,"Gpu ID")
cmd:option("-manualSeed",2 ,"manualSeed for Random generator")
params = cmd:parse(arg)

-----------------------------------------------------------------------------
--------------------- Initialize Variable -----------------------------------
-----------------------------------------------------------------------------
if params.log ~= "" then
   cmd:log(params.log, params)
   cmd:addTime('torch_benchmarks','%F %T')
   print("setting log file as "..params.log)
end

libs = {}
run_on_cuda = false
if params.gpu then
    print("using cudnn")
    require 'cudnn'
    libs['SpatialConvolution'] = cudnn.SpatialConvolution
    libs['SpatialMaxPooling'] = cudnn.SpatialMaxPooling
    libs['ReLU'] = cudnn.ReLU
    torch.setdefaulttensortype('torch.CudaTensor')
    run_on_cuda = true
else
    libs['SpatialConvolution'] = nn.SpatialConvolution
    libs['SpatialMaxPooling'] = nn.SpatialMaxPooling
    libs['ReLU'] = nn.ReLU
    torch.setdefaulttensortype('torch.FloatTensor')
end

epoch = 0
batch_size = params.batch_size
--Load model and criterion

if params.weights ~= "" then
    print("loading model from pretained weights in file "..params.weights)
    model = torch.load(params.weights)
else
    model = build_model(libs)
end

print(model)
criterion = nn.CrossEntropyCriterion()

if run_on_cuda then
    cutorch.setDevice(params.GPU) -- by default, use GPU 1
    model = model:cuda()
    criterion = criterion:cuda()
end

torch.manualSeed(params.manualSeed)

-----------------------------------------------------------------------------
--------------------- Training Function -------------------------------------
-----------------------------------------------------------------------------
-- retrieve a view (same memory) of the parameters and gradients of these (wrt loss) of the model (Global)
parameters, grad_parameters = model:getParameters();

function train(traindata, testdata)
    local saved_criterion = false;
    for i = 1, params.max_epochs do
        
        train_one_epoch(traindata)
        
        if i % params.snapshot_epoch == 0 then
           test_one_epoch(testdata) 
        end    
            
        if params.snapshot_epoch > 0 and (epoch % params.snapshot_epoch) == 0 then -- epoch is global (gotta love lua :p)
            local filename = paths.concat(params.snapshot_dir, "snapshot_model.net")
            os.execute('mkdir -p ' .. sys.dirname(filename))
            torch.save(filename, model)        
            --must save std, mean and criterion?
            if not saved_criterion then
                local criterion_filename = paths.concat(params.snapshot_dir, "_criterion.net")
                torch.save(criterion_filename, criterion)
                local dataset_attributes_filename = paths.concat(params.snapshot_dir, "_dataset.params")
                --dataset_attributes = {}
                --dataset_attributes.mean = triandata.mean
                --dataset_attributes.std = traindata.std
                --torch.save(dataset_attributes_filename, dataset_attributes)
            end
            print("Checkpoint saved!")
        end
    end
end

-- GPU inputs (preallocate)
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

-- Permutations
local permutations = torch.load(params.shuffle)	--Ananth

function train_one_epoch(dataset)
    collectgarbage()
    model:cuda()
    model:training()
    local time = sys.clock()
    --train one epoch of the dataset
    local avg_error = 0 -- the average error of all criterion outs
    for mini_batch_start = 1, dataset:size(), batch_size do --for each mini-batch
        
        local _start = mini_batch_start
        --local _end = math.min(mini_batch_start + batch_size - 1, dataset:size())
        --local _end = ( (mini_batch_start + batch_size - 1) < dataset:size() ) and mini_batch_start + batch_size - 1 or dataset:size()
        local _end = ( (mini_batch_start + batch_size - 1) < dataset:size() ) and batch_size or ( dataset:size() - mini_batch_start)

        --local dd = dataset.data:read('/images'):partial({_start,_end},{1,1},{1,255},{1,255})
        --inputs_permuted, labels_permuted = createPuzzle(dd)
        inputs_permuted, labels_permuted = createPuzzle(dataset.data:narrow(1,_start, _end))
        
        --torch.save("temp_inp.t7",inputs_permuted)
        --torch.save("temp_lab.t7",labels_permuted)
        
        -- shift everything to GPU
        inputs:resize(inputs_permuted:size()):copy(inputs_permuted)
        labels:resize(labels_permuted:size()):copy(labels_permuted)
        --create a closure to evaluate df/dX where x are the model parameters at a given point
        --and df/dx is the gradient of the loss wrt to thes parameters

        local func_eval = 
        function(x)
                --update the model parameters (copy x in to parameters)
                if x ~= parameters then
                    parameters:copy(x) 
                end
                grad_parameters:zero() --reset gradients


                local output = model:forward(inputs)   -- (inputs[i])
                local err = criterion:forward(output, labels)         --(output, labels[i])
                avg_error = avg_error + err

                --estimate dLoss/dW
                local dloss_dout = criterion:backward(output, labels)  --(output, labels[i])
                model:backward(inputs, dloss_dout)     -- (inputs[i], dloss_dout)

                grad_parameters:div(inputs:size(1));
            
                print("[" .. epoch .. "] avg_error " .. (err / inputs:size(1)) .. " " .. "[" .. mini_batch_start .. "/" .. dataset:size() .."]")
            
                return (err / inputs:size(1)), grad_parameters
        end


        config = {learningRate = params.learning_rate, momentum = params.momentum}

        --This function updates the global parameters variable (which is a view on the models parameters)
        optim.sgd(func_eval, parameters, config)
        
        --xlua.progress(mini_batch_start, dataset:size()) --display progress
    end

    -- time taken
    time = sys.clock() - time
    print("time taken for 1 epoch = " .. (time * 1000) .. "ms, time taken to learn 1 sample = " .. ((time/dataset:size())*1000) .. 'ms' .. ' evg_error' .. avg_error / dataset:size())
 
    epoch = epoch + 1
end
function test_one_epoch(dataset)
    collectgarbage()
    --print(#dataset)
    model:cuda()
    model:evaluate()
    local time = sys.clock()
    --train one epoch of the dataset
    local outputs = torch.CudaTensor() 
    local labels_permuted_all = torch.CudaTensor()
    local avg_error = 0 -- the average error of all criterion outs

    for mini_batch_start = 1, dataset:size(), batch_size do --for each mini-batch
        
        local _start = mini_batch_start
        --local _end = math.min(mini_batch_start + batch_size - 1, dataset:size())
        --local _end = ( (mini_batch_start + batch_size - 1) < dataset:size() ) and mini_batch_start + batch_size - 1  or dataset:size()
        local _end = ( (mini_batch_start + batch_size - 1) < dataset:size() ) and batch_size or ( dataset:size() - mini_batch_start)
        
        inputs_permuted, labels_permuted = createPuzzle(dataset.data:narrow(1,_start, _end))
        --local dd = dataset.data:read('/images'):partial({_start,_end},{1,1},{1,255},{1,255})
        --inputs_permuted, labels_permuted = createPuzzle(dd)
        
        -- shift everything to GPU
        inputs:resize(inputs_permuted:size()):copy(inputs_permuted)
        labels:resize(labels_permuted:size()):copy(labels_permuted)
	
        local output = model:forward(inputs)   -- (inputs[i])
         
        outputs = (mini_batch_start== 1) and output or torch.cat(outputs,output,1)
        labels_permuted_all = (mini_batch_start==1) and labels or torch.cat(labels_permuted_all,labels,1)
	local err = criterion:forward(output, labels)         --(output, labels[i])
	avg_error = avg_error + err;
        
        print("TEST " .. _end  .. " error: " .. err/inputs:size(1) .. " [" .. mini_batch_start .. "/" .. dataset:size() .. "]" )

        --xlua.progress(mini_batch_start, dataset:size()) --display progress
    end
    
    local top1_center = 0    
    local pred = outputs:float()
    
    local _, pred_sorted = pred:sort(2, true)

    --torch.save("tempPredictions.t7", pred:double())
    --torch.save("templabels.t7",labels_permuted_all:double())

    for i=1,pred_sorted:size(1) do
       local g = labels_permuted_all[i][1]
       if pred_sorted[i][1] == g then top1_center = top1_center + 1 end
    end
    print("TEST avg_error " .. avg_error/dataset:size() .. " top-1 " .. top1_center*100/dataset:size())
    
    -- time taken
    time = sys.clock() - time
    print("time taken for 1 epoch = " .. (time * 1000) .. "ms, time taken to learn 1 sample = " .. ((time/dataset:size())*1000) .. 'ms' .. " avg_error: " .. avg_error/dataset:size() .. " top-1 " .. top1_center)

end

function _test(dataset)
    collectgarbage()
    model:cuda()
    model:testing()
    local time = sys.clock()
    --train one epoch of the dataset

    inputs_permuted, labels_permuted = createPuzzle(dataset.data)
        
    -- shift everything to GPU
    inputs:resize(inputs_permuted:size()):copy(inputs_permuted)
    labels:resize(labels_permuted:size()):copy(labels_permuted)


    local avg_error = 0 -- the average error of all criterion outs
    outputs = torch.CudaTensor()
    --evaluate for complete mini_batch
    local output = model:forward(inputs[i])
    outputs = (i == 1) and output or torch.cat(outputs,output,2)
    local err = criterion:forward(output, labels[i])
    avg_error = avg_error + err
        
    avg_error = avg_error / inputs:size(1);

    local pred = outputs:t():float()
    local _, pred_sorted = pred:sort(2, true)
    for i=1,pred:size(1) do
       local g = labels[i]
       if pred_sorted[i][1] == g then top1_center = top1_center + 1 end
    end
    print("test avg_error " .. avg_error .. " top-1 " .. top1_center)        
    xlua.progress(1, dataset:size()) --display progress
    -- time taken
    time = sys.clock() - time
    print("time taken for 1 epoch = " .. (time * 1000) .. "ms, time taken to learn 1 sample = " .. ((time/dataset:size())*1000) .. 'ms')
    epoch = epoch + 1
end

function _createPuzzle(input_images)

   local sH = 64
   local sW = 64
   local nc = 3		--channels
   local np = 4		-- patches 3X3
   local nim = input_images:size(1)	--Number of images 
   
   local puzzles = torch.Tensor(nim,np,nc,sW,sH)
   local labels = torch.Tensor(nim,1)
   for ii = 1,nim do   
      local im = input_images[ii]
      --im = torch.cat({im,im,im},1)
      
      local patches = torch.Tensor(np,nc,sW,sH)
      patches[1] = torch.cat({im[1]:narrow(1,1,64):narrow(2,1,64),im[1]:narrow(1,1,64):narrow(2,1,64),im[1]:narrow(1,1,64):narrow(2,1,64)},1)

      patches[2] = torch.cat({im[1]:narrow(1,1,64):narrow(2,64,64),im[1]:narrow(1,1,64):narrow(2,64,64),im[1]:narrow(1,1,64):narrow(2,64,64)},1)

      patches[3] = torch.cat({im[1]:narrow(1,64,64):narrow(2,1,64),im[1]:narrow(1,64,64):narrow(2,1,64),im[1]:narrow(1,64,64):narrow(2,1,64)},1)
      
      patches[4] = torch.cat({im[1]:narrow(1,64,64):narrow(2,64,64), im[1]:narrow(1,64,64):narrow(2,64,64), im[1]:narrow(1,64,64):narrow(2,64,64)},1)


      -- Take a random number between 1 and 100	
      -- This rand_index will be the label
      local rand_index = torch.random(1,24)
      local perm = permutations[rand_index]	-- get the permutation

      local puzzle = patches:index(1,perm:long()) 	-- shuffle the images

      -- finally, copy the puzzle and labels [random index]
      puzzles[ii] = puzzle
      labels[ii] = rand_index

   end
   return puzzles, labels

end

-- Create puzzles Ananth
function createPuzzle(input_images)

   local oH = 225
   local oW = 225
   local sH = 64
   local sW = 64
   local nc = 3		--channels
   local np = 9		-- patches 3X3
   local nim = input_images:size(1)	--Number of images 
   local puzzles = torch.Tensor(nim,np,nc,sW,sH)
   local labels = torch.Tensor(nim,1)
   print()
   for ii = 1,nim do   
      local im = input_images[ii]
      im = torch.cat({im,im,im},1)
        
      local iW = im:size(3)
      local iH = im:size(2)

      --local h1 = math.ceil(torch.uniform(1e-2, iH-oH))
      --local w1 = math.ceil(torch.uniform(1e-2, iW-oW))

      local patches = torch.Tensor(np,nc,sW,sH)

      local count = 1
      for w1=0,oW-oW/3,oW/3 do
           for h1=0,oH-oH/3,oH/3 do  
              w2 = w1 + math.ceil(torch.uniform(1e-2, oW/3-sW))
              h2 = h1 + math.ceil(torch.uniform(1e-2, oH/3-sH))
              patches[count] = image.crop(im, w2, h2, w2 + sW, h2 + sH)
              count = count + 1
    	   end
      end

      -- Take a random number between 1 and 100	
      -- This rand_index will be the label
      local rand_index = torch.random(1,100)
      local perm = permutations[rand_index]	-- get the permutation

      local puzzle = patches:index(1,perm:long()) 	-- shuffle the images

      -- finally, copy the puzzle and labels [random index]
      puzzles[ii] = puzzle
      labels[ii] = rand_index

   end
   
   return puzzles, labels
end

-----------------------------------------------------------------------------
--------------------- Training Function -------------------------------------
-----------------------------------------------------------------------------
print("loading dataset...")

local train_dataset = mnist.load_siamese_dataset(params.train_data)
local test_dataset = mnist.load_siamese_dataset(params.val_data)

--train_dataset = mnist.load_siamese_dataset('train_4.t7')
--test_dataset = mnist.load_siamese_dataset('val_4.t7')

--train_dataset = mnist.load_siamese_dataset('temp.t7')
--test_dataset = mnist.load_siamese_dataset('temp.t7')

print("dataset loaded")
train(train_dataset,test_dataset)
