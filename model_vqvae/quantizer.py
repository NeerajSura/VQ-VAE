import torch
import torch.nn as nn
from einops import einsum


class Quantizer(nn.Module):
    def __init__(self, config):
        super(Quantizer, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(num_embeddings=config['codebook_size'], embedding_dim=config['latent_dim'])


    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1) # B, H, W, C
        x = x.reshape(x.size(0), -1, x.size(-1)) # (B, H*W, C)
        #print(f'x shape is {x.shape}')
        
        self.embedding.weight[None, :].repeat((x.size(0), 1, 1))
        #print(f'embeddings dimension is {self.embedding.weight[None, :].repeat((x.size(0), 1, 1)).shape}')
        dist = torch.cdist(x, self.embedding.weight[None, :].repeat((x.size(0), 1, 1))) # B, codebook_size, latent_dim

        min_encoding_indices = torch.argmin(dist, dim=-1)

        quant_out = torch.index_select(self.embedding.weight, 0, min_encoding_indices.view(-1))

        x = x.reshape((-1, x.size(-1)))
        commitment_loss = torch.mean((quant_out.detach() - x) ** 2)
        codebook_loss = torch.mean((x.detach() - quant_out) ** 2)
        quantize_losses = {
            'commitment_loss' : commitment_loss,
            'codebook_loss' : codebook_loss
        }

        quant_out = x + (quant_out - x).detach()
        quant_out = quant_out.reshape((B, C, H, W)).permute(0, 3, 1, 2)
        min_encoding_indices = min_encoding_indices.reshape((-1, quant_out.size(-2), quant_out.size(-1)))
        #print(f'quant_out shape is {quant_out.shape}')
        return quant_out, quantize_losses, min_encoding_indices
        
    def quantize_indices(self, indices):
        return einsum(indices, self.embedding.weight, 'b n h w, n d -> b d h w')


def get_quantizer(config):
    quantizer = Quantizer(config = config['model_params'])

    return quantizer
