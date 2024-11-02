import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config
        activation_map = {
            'relu': nn.ReLU(),
            'leaky': nn.LeakyReLU(),
            'tanh': nn.Tanh(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU()
        }

        # Validate the config for the model_vqvae is correctly setup
        assert config['conv_activation_fn'] is None or config['conv_activation_fn'] in activation_map
        self.latent_dim = config['latent_dim']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Decoder ConvolutionTranspose + bn + activation fn
        self.decoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(in_channels=config['transposebn_channels'][i], out_channels=config['transposebn_channels'][i+1], kernel_size=config['transpose_kernel_size'][i],
                                   stride=config['transpose_kernel_strides'][i], padding=0),
                nn.BatchNorm2d(config['transposebn_channels'][i+1]),
                activation_map[config['transpose_activation_fn']]
            ) for i in range(config['transpose_bn_blocks']-1)
        ])

        dec_last_idx = config['transpose_bn_blocks']

        self.decoder_layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(config['transposebn_channels'][dec_last_idx - 1], config['transposebn_channels'][dec_last_idx],
                                kernel_size=config['transpose_kernel_size'][dec_last_idx - 1],
                                stride=config['transpose_kernel_strides'][dec_last_idx - 1],
                                padding=0),
                nn.Tanh()
            )
        )


    def forward(self, x):
        out = x
        for layer in self.decoder_layers:
            out = layer(out)
        return out


def get_decoder(config):
    decoder = Decoder(config = config['model_params'])
    return decoder