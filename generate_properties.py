
import sys
import numpy as np
import onnxruntime as onnxrun
import csv

import torch
import torchvision.datasets as dset
import torchvision.transforms as trans
from torch.utils.data import DataLoader
from torch.utils.data import sampler

# Force download of MNIST from aws mirror.
dset.MNIST.mirrors = [
  # "http://yann.lecun.com/exdb/mnist/",
  "https://ossci-datasets.s3.amazonaws.com/mnist/",
]

def obtainTestImgs(nets,Xs,Ts,numImgs):
  # Create onnx-runtime sessions.
  sess = [onnxrun.InferenceSession(net) for net in nets]

  # Store selected images and targets.
  selectedIds = []
  selectedXs = []
  selectedTs = []

  for i in range(0,len(Xs)):
    # Extract an input and the corresponding target.
    xi = Xs[i]
    ti = Ts[i]
    # Compute predictions.
    yis = [np.argmax(s.run(None, {'input': xi.numpy()})) for s in sess]

    if all(yis == ti.numpy()):
      # All networks predict the correct result. We have a candidate image.
      selectedIds.append(i)
      selectedXs.append(xi)
      selectedTs.append(ti)
      # Check if we have found enough images.
      if len(selectedIds) >= numImgs:
        return selectedIds, selectedXs, selectedTs

def writeVnnlib(filename,instName,nIn,nOut,xl,xu,t):
  # Create the file.
  with open(filename,'w') as f:
    # In the first line write the name of the instance.
    f.write(f'; {instName}\n')

    # Declare the input variables.
    f.write('; Declare the input variables.\n')
    for i in range(0,nIn):
      f.write(f'(declare-const X_{i} Real)\n')
    # Declare the output variables.
    f.write('; Declare the output variables.\n')
    for i in range(0,nOut):
      f.write(f'(declare-const Y_{i} Real)\n')

    # Write the input constraints.
    f.write('; Input constraints.\n')
    for i in range(0,nIn):
      f.write(f'(assert (<= X_{i} {xu[0,i]}))\n')
      f.write(f'(assert (>= X_{i} {xl[0,i]}))\n')

    # Write the output constraints.
    f.write('; Output constraints.\n')
    f.write('(assert (or\n')
    for i in range(0,nOut):
      if i != t:
        f.write(f'    (and (>= Y_{i} Y_{t.numpy()}))\n')
    f.write('))')

def loadDataset(datasetName):
  # Load dataset and set corresponding perturbation radius.
  if ds == 'mnist':
    # Load the dataset.
    dataset = dset.MNIST('./tmp',train=False,download=True,transform=trans.ToTensor())
    # Set perturbation radius.
    epsilon = 0.1
  elif ds == 'svhn':
    # Load the dataset.
    dataset = dset.SVHN('./tmp',split='test',download=True,transform=trans.ToTensor())
    # Set perturbation radius.
    epsilon = 0.1
  else: # ds == 'cifar10':
    # Load the dataset.
    dataset = dset.CIFAR10('./tmp',train=False,download=True,transform=trans.ToTensor())
    # Set perturbation radius.
    epsilon = 2/255

  loader = DataLoader(dataset,batch_size=10000,sampler=sampler.SubsetRandomSampler(range(10000)))
  # Instantiate iterator.
  Xs, Ts = next(iter(loader))

  # Reshape inputs to vectors.
  Xs = np.transpose(Xs,(0,3,2,1)).reshape(len(Xs),1,-1)

  return Xs, Ts, epsilon

if __name__ == '__main__':
    seed = 1
    # Replace seed if there is one.
    if len(sys.argv) == 2:
        seed = int(sys.argv[1])
    # Set torch seed.
    torch.random.manual_seed(seed)

    # Path to networks.
    nets = lambda dsname: [f'./nns/{dsname}-point.onnx',f'./nns/{dsname}-trades.onnx',f'./nns/{dsname}-set.onnx']

    # Specify path for .vnnlib files
    savePath = f'./benchmark-files'

    # For each network select 20 random input.
    numTestImgs = 20

    # Specify timeout [in seconds].
    timeout = 30

    # Initialize instances.
    instances = []
    for ds in ['mnist','svhn','cifar10']:
      print(f"Generating instances for '{ds}'",end='... ')
      # Load dataset and set corresponding perturbation radius.
      Xs, Ts, epsilon = loadDataset(ds)
      # Obtain number of input and output dimensions.
      nIn = Xs.shape[2]
      nOut = len(np.unique(Ts))
      # Find images that are correctly classified by all nets.
      selectedIds, selectedXs, selectedTs = obtainTestImgs(nets(ds),Xs,Ts,numTestImgs)

      for i in range(0,len(selectedIds)):
        # Extract selected image and target.
        id = selectedIds[i]
        xi = selectedXs[i]
        ti = selectedTs[i]
        # Compute input bounds; restricted to [0,1].
        xil = torch.clip(xi - epsilon,0,1)
        xiu = torch.clip(xi + epsilon,0,1)
        # Construct filename
        instName = f'{ds}-img{id}.vnnlib'
        filename = f'{savePath}/{instName}'
        writeVnnlib(filename,instName,nIn,nOut,xil,xiu,ti)
        # Store instances.
        [instances.append([net,filename,timeout]) for net in nets(ds)]

      print(f'done')

    # Create instances.csv file.
    with open(f'{savePath}/instances.csv','w',newline='') as instCsv:
      writer = csv.writer(instCsv)
      for instance in instances:
        writer.writerow(instance)
    