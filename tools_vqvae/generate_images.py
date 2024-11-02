import argparse

import numpy as np
import yaml
import os
import torch
import pickle
from tqdm import tqdm
import torchvision
from model_vqvae.vqvae import get_model
from torchvision.utils import make_grid
from tools_vqvae.train_lstm import MnistLSTM
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate(args):
    r"""
    Sampling in VqVae
    1. Create config
    2. Create and load VqVae model
    3. Create and load LSTM model
    4. Generate 100 encoder outputs from trained LSTM
    5. pass them to the trained decoder
    6. Save the images
    :param args:
    :return:
    """

    # Create config
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exp:
            print(exp)

    # Create and load the VQVAE model checkpoint and set to eval mode
    vqvae_model = get_model(config).to(device)
    vqvae_model = vqvae_model.to(device)
    assert os.path.exists(os.path.join(config['train_params']['task_name'],
                                       config['train_params']['ckpt_name'])), "Train the vqvae model first"
    vqvae_model.load_state_dict(torch.load(os.path.join(config['train_params']['task_name'],
                                                        config['train_params']['ckpt_name']), map_location=device))

    vqvae_model.eval()

    # Create and load the LSTM model and set to eval mode
    default_lstm_config = {
        'input_size' : 2,
        'hidden_size' : 128,
        'codebook_size' : config['model_params']['codebook_size']
    }

    model = MnistLSTM(input_size=default_lstm_config['input_size'], hidden_size=default_lstm_config['hidden_size'],
                      codebook_size=default_lstm_config['codebook_size']).to(device)
    model.to(device)
    assert os.path.exists(os.path.join(config['train_params']['task_name'],
                                                    'best_mnist_lstm.pth')), "Train the lstm first"
    model.load_state_dict(torch.load(os.path.join(config['train_params']['task_name'],
                                                    'best_mnist_lstm.pth'), map_location=device))
    model.eval()

    # Generate 100 outputs from trained LSTM
    generated_quantized_indices = []
    mnist_encodings = pickle.load(open(os.path.join(config['train_params']['task_name'],
                                                    config['train_params']['output_train_dir'],
                                                    'mnist_encodings.pkl'), 'rb'))
    mnist_encodings_length = mnist_encodings.reshape(mnist_encodings.size(0), -1).shape[-1]
    context_size = 32
    num_samples = 100
    print('Generating Samples')

    for _ in tqdm(range(num_samples)):
        # Initialize the start token
        ctx = torch.ones((1)).to(device) * config['model_params']['codebook_size']

        for i in range(mnist_encodings_length):
            padded_ctx = ctx

            if len(ctx) < context_size:
                # Pad context with pad token
                padded_ctx = torch.nn.functional.pad(padded_ctx, (0, context_size - len(ctx)), "constant",
                                                     config['model_params']['codebook_size'] + 1)
            out = model(padded_ctx[None, :].long().to(device))
            probs = torch.nn.functional.softmax(out, dim=-1)
            pred = torch.multinomial(probs[0], num_samples=1)

            # Update the ctx with new prediction
            ctx = torch.cat([ctx, pred])
        generated_quantized_indices.append(ctx[1:][None, :])

    ######## Decode the Generated Indices ##########
    generated_quantized_indices = torch.cat(generated_quantized_indices, dim=0)
    h = int(generated_quantized_indices[0].size(-1) ** 0.5)
    quantized_indices = generated_quantized_indices.reshape((generated_quantized_indices.size(0), h, h)).long()
    quantized_indices = torch.nn.functional.one_hot(quantized_indices, config['model_params']['codebook_size'])
    quantized_indices = quantized_indices.permute((0, 3, 1, 2))
    output = vqvae_model.decode_from_codebook_indices(quantized_indices.float())

    # Transform from -1, 1 range to 0,1
    output = (output + 1) / 2

    if config['model_params']['in_channels'] == 3:
        # Just because we took input as cv2.imread which is BGR so make it RGB
        output = output[:, [2, 1, 0], :, :]
    grid = make_grid(output.detach().cpu(), nrow=10)

    img = torchvision.transforms.ToPILImage()(grid)
    img.save(os.path.join(config['train_params']['task_name'],
                          config['train_params']['output_train_dir'],
                          'generation_results.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for lstm generation')
    parser.add_argument('--config', dest='config_path', default='config/vqvae_colored_mnist.yaml', type=str)
    args = parser.parse_args()
    generate(args)

