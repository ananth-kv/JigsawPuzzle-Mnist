require 'nn'
require 'loadcaffe'

function build_model(libs)
    
   return torch.load'models/lenet_siamese.dat'
   --return torch.load('models/model.dat')
   --return torch.load('models/model_4.t7')    
end

function __build_model(libs)
   print("Builing model....")
   local siamese_raw = loadcaffe.load('caffeFiles/deploy_cfn_jps.prototxt','caffeFiles/cfn_jps.caffemodel','cudnn')
   
   local features = nn.Sequential()

   for i=1,19 do
      features:add(siamese_raw.modules[i])
   end
    
   siamese_encoder = nn.ParallelTable()
   siamese_encoder:add(features)

   for i =1,9 do
      siamese_encoder:add(features:clone('weight','bias', 'gradWeight','gradBias'))
   end

   -- 1.4. Combine 1.1 and 1.3 to produce final classifier
   local classifier = nn.Sequential()
   classifier:add(nn.JoinTable(1))
   classifier:add(nn.Linear(9*1024,4096))
   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(4096,2400))
   classifier:add(nn.Linear(2400,100))
   classifier:add(nn.LogSoftMax())

   local model = nn.Sequential()
   model:add(nn.SplitTable(1))
   model:add(siamese_encoder)
   model:add(classifier)

   print("done...")
   return model

end

function _build_model(libs)
    local SpatialConvolution = libs['SpatialConvolution']
    local SpatialMaxPooling = libs['SpatialMaxPooling']
    local ReLU = libs['ReLU']
    --Encoder/Embedding
    --Input dims are 28x28          NOTE: change dims as inputs are 32x32 -- Need to do this 
    encoder = nn.Sequential()
    encoder:add(nn.SpatialConvolution(1, 20, 5, 5)) -- 1 input image channel, 20 output channels, 5x5 convolution kernel (each feature map is 28x28)
    encoder:add(nn.SpatialMaxPooling(2, 2, 2, 2)) -- max pooling with kernal 2x2 and a stride of 2 in each direction (feature maps are 14x14)
    encoder:add(nn.SpatialConvolution(20, 50, 5, 5)) -- 20 input feature maps and output 50, 5x5 convolution kernel (feature maps are 10x10) 
    encoder:add(nn.SpatialMaxPooling(2, 2, 2, 2)) -- max pooling (feature maps are 5x5)

    encoder:add(nn.View(50*5*5)) --reshapes to view data at 50x4x4
    encoder:add(nn.Linear(50*5*5, 500))
    encoder:add(nn.ReLU())
    encoder:add(nn.Linear(500, 10))
    --encoding layer - go from 10 class out to 2-dimensional encoding
    encoder:add(nn.Linear(10, 2))

    --The siamese model
    siamese_encoder = nn.ParallelTable()
    siamese_encoder:add(encoder)
    siamese_encoder:add(encoder:clone('weight','bias', 'gradWeight','gradBias')) --clone the encoder and share the weight, bias. Must also share the gradWeight and gradBias


    --The siamese model (inputs will be Tensors of shape (2, channel, height, width))
    model = nn.Sequential()
    model:add(nn.SplitTable(1)) -- split input tensor along the rows (1st dimension) to table for input to ParallelTable
    model:add(siamese_encoder)
    model:add(nn.PairwiseDistance(2)) --L2 pariwise distance

    margin = 1
    criterion = nn.HingeEmbeddingCriterion(margin)
    return model
end


