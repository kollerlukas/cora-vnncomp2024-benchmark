% Load neural network.
nn = neuralNetwork.readONNXNetwork('./nns/cifar10-set.onnx');

% Load specification.
[X0,specs] = vnnlib2cora('benchmark-files/cifar10-img353.vnnlib');

%%

% Measure verification time.
tic

% Instantiate input zonotope.
c = 1/2*(X0{1}.sup + X0{1}.inf);
G = diag(1/2*(X0{1}.sup - X0{1}.inf));
Z = zonotope(c,G);

% Create evaluation options.
options.nn = struct(...
    'use_approx_error',true,...
    'poly_method','bounds'...
);
% Set default training parameters
options = nnHelper.validateNNoptions(options,true);

% In each layer, store ids of active generators and identity matrices 
% for fast adding of approximation errors.
nn.prepareForZonoBatchEval(c,options);

% Compute output.
Y = nn.evaluateZonotopeBatch(c,G,options);

% Check specification.
if all(interval(specs.set.A*Y).sup <= 0)
    fprintf('VERIFIED\n');
else
    fprintf('UNKNOWN\n');
end

time = toc;

fprintf('Verification time: %.4f [s]\n',time);
