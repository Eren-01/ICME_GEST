from fourier_module import Fourier2d
from graphode_module import SGODEModel
import torch
import torch.nn as nn

def simple_logger(message):
        print(message)

class MainModel(nn.Module):
    def __init__(self, fourier2d_config, sgode_config, device='cpu'):
        super(MainModel, self).__init__()
        self.device = device
        self.fourier2d = Fourier2d(**fourier2d_config).to(self.device)
        self.sgode_model = SGODEModel(**sgode_config).to(self.device)

    def forward(self, inputs, labels, batches_seen=0):
        inputs_sgode = inputs_fourier2d = inputs.to(self.device)
        output_fourier2d = self.fourier2d(inputs_fourier2d)
        # output_sgode = self.sgode_model.forward('without_regularization', inputs_sgode, labels, batches_seen)
        # combined_output = output_fourier2d + output_sgode
        combined_output = output_fourier2d
        return combined_output




if __name__ == "__main__":
    fourier2d_config = {
        'modes1': 12,
        'modes2': 12,
        'pred_len': 10,
        'width': 20,
    }

    sgode_config = {
        'input_dim': 1, 
        'seq_len': 10, 
        'horizon': 10,  
        'num_nodes': 4096,  
        'rnn_units': 64, 
        'embed_dim': 10,
        'Atype': 2,  
        'max_diffusion_step': 2,  
        'cl_decay_steps': 1000, 
        'use_ode_for_gru': True, 
        'filter_type': 'laplacian',
        'logger': simple_logger,
        'temperature': 1.0,
    }


    # device = torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seq_len = 10  
    batch_size = 5
    heigth = 64
    width = 64
    inputs = torch.randn(batch_size, heigth, width, seq_len).to(device)
    labels =  torch.randn(batch_size, heigth, width, seq_len).to(device)
    print("inputs shape:", inputs.shape)
    print("labels shape:", labels.shape)
    batches_seen = 100
    main_model = MainModel(fourier2d_config, sgode_config, device=device)
    outputs = main_model(inputs, labels)
    print("pred shape:", outputs.shape)

