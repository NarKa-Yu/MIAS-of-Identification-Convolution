## MIAs of identification convolution

Please note: 
1. This experiment does not include downloaded datasets. Please set up an additional folder 'dataset' to place the corresponding dataset and manually modify the dataset call.
2. Please manually modify the selection of models and datasets.

- Requirement:
  - python3.8
  - torch
  - torchvision

Enter the following command in the terminal 
to execute membership inference attack:</br>
`python attack.py fed=sgd idx=0 amount=10 scaling=100000 epoch=10 batch_size=64 ic=midrange region=var`
