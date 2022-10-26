import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda'
is_cuda = device == 'cuda'
