import argparse
import json
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformer_mpc_vit import CustomFlaxViTForImageClassification
from transformers.models.vit.configuration_vit import ViTConfig
from dataloader_tinyimagenet import TinyImageNet
import flax
from flax import traverse_util
import jax.numpy as jnp
import numpy as np
import spu.utils.distributed as ppd
import spu.spu_pb2 as spu_pb2
parser = argparse.ArgumentParser(description='distributed driver.')

parser.add_argument(
    "-c", "--config", default="2pc.json"
)
args = parser.parse_args()
with open(args.config, 'r') as file:
    conf = json.load(file)

ppd.init(conf["nodes"], conf["devices"])

# below two functions just change the pytorch layer names to their corresponding flax layer names and loads them into a flax model
def map_qkv_weights_to_flax(pytorch_qkv_weight):
    """
    This function assumes pytorch_qkv_weight has shape (C, 3 * C).
    It splits this into three (C, C) matrices for query, key, and value.
    """
    C = pytorch_qkv_weight.shape[0]
    third_dim = C // 3  # Assuming 3 for q, k, v (576,192)

    query_weight = pytorch_qkv_weight[:third_dim,:]
    key_weight = pytorch_qkv_weight[third_dim : 2 * third_dim,:]
    value_weight = pytorch_qkv_weight[2 * third_dim:,:]
    
    return query_weight, key_weight, value_weight

def pytorch_to_flax(pytorch_model_state_dict, flax_model):
    flax_params = {}
    for name, param in pytorch_model_state_dict.items():
        if 'conv' in name:
            if param.dim() == 4:
                param_data = param.detach().cpu().numpy()
                param_data = jnp.transpose(param_data,axes=(2,3,1,0))
            elif param.dim() == 1:  # Typically bias
                param_data = param.detach().cpu().numpy()
            else:
                print(f"Unhandled shape {param.shape} for {name}")
                continue
        else:
            param_data = param.detach().cpu().numpy()
        
        name_parts = name.split('.')
        
        if len(name_parts) > 3:
            if name_parts[0] == "classifier" and name_parts[1] == "blocks" and name_parts[3] == "pre_norm":
                if name_parts[4] == "weight":
                    flax_name = tuple(("vit", "encoder", "layer", name_parts[2], 'layernorm_before',"scale"))
                    flax_params[flax_name] = param_data
                elif name_parts[4] == "bias":
                    flax_name = tuple(("vit", "encoder", "layer", name_parts[2], 'layernorm_before',"bias")) # flax 
                    flax_params[flax_name] = param_data
            
            if name_parts[0] == "classifier" and name_parts[1] == "blocks" and name_parts[3] == "norm1":
                if name_parts[4] == "weight":
                    flax_name = tuple(("vit", "encoder", "layer", name_parts[2], 'layernorm_after',"scale"))

                    flax_params[flax_name] = param_data
                else:
                    flax_name = tuple(("vit", "encoder", "layer", name_parts[2], 'layernorm_after',"bias"))

                    flax_params[flax_name] = param_data
            
            if name_parts[0] == "classifier" and name_parts[1] == "blocks" and name_parts[3] == "self_attn" and name_parts[4] == "qkv":
                weight = param.data.cpu().numpy()
                query_weight, key_weight, value_weight = map_qkv_weights_to_flax(weight)
                flax_name_key = tuple(("vit", "encoder", "layer", name_parts[2], 'attention','attention','key',"kernel"))
                flax_name_query = tuple(("vit", "encoder", "layer", name_parts[2], 'attention','attention','query',"kernel"))
                flax_name_value = tuple(("vit", "encoder", "layer", name_parts[2], 'attention','attention','value',"kernel"))
                flax_params[flax_name_key] = jnp.transpose(key_weight)
                flax_params[flax_name_query] = jnp.transpose(query_weight)
                flax_params[flax_name_value] = jnp.transpose(value_weight)
            
            if name_parts[0] == "classifier" and name_parts[1] == "blocks" and name_parts[3] == "self_attn" and name_parts[4] == "alpha":
                flax_name = tuple(('vit', 'encoder', 'layer', name_parts[2], 'attention', 'attention', 'alpha'))
                flax_params[flax_name] = param_data
            if name_parts[0] == "classifier" and name_parts[1] == "blocks" and name_parts[3] == "self_attn" and name_parts[4] == "proj":
                if name_parts[5] == "weight":
                    flax_name = tuple(('vit', 'encoder', 'layer',name_parts[2],'attention', 'output', 'dense', 'kernel'))
                    flax_params[flax_name] = jnp.transpose(param_data,(1,0))
                else:
                    flax_name = tuple(('vit', 'encoder', 'layer',name_parts[2],'attention', 'output', 'dense', 'bias'))
                    flax_params[flax_name] = param_data
                

            if name_parts[0] == "classifier" and name_parts[1] == "blocks" and name_parts[3] == "linear1" and name_parts[4] == "weight":
                flax_name = tuple(("vit", "encoder", "layer", name_parts[2], 'intermediate','dense','kernel'))
                flax_params[flax_name] = jnp.transpose(param_data,(1,0))
            
            if name_parts[0] == "classifier" and name_parts[1] == "blocks" and name_parts[3] == "linear1" and name_parts[4] == "bias":
                flax_name = tuple(("vit", "encoder", "layer", name_parts[2], 'intermediate','dense','bias'))
                flax_params[flax_name] = param_data

            if name_parts[0] == "classifier" and name_parts[1] == "blocks" and name_parts[3] == "linear2" and name_parts[4] == "weight":
                flax_name = tuple(("vit", "encoder", "layer", name_parts[2], 'output','dense','kernel'))
                flax_params[flax_name] = jnp.transpose(param_data)
            
            if name_parts[0] == "classifier" and name_parts[1] == "blocks" and name_parts[3] == "linear2" and name_parts[4] == "bias":
                flax_name = tuple(("vit", "encoder", "layer", name_parts[2], 'output','dense','bias'))
                flax_params[flax_name] = param_data
        
        if name_parts[0] == "classifier" and name_parts[1] == "class_emb":
            flax_name = tuple(('vit', 'embeddings', 'cls_token'))
            flax_params[flax_name] = param_data
        
        if name_parts[0] == "classifier" and name_parts[1] == "positional_emb":
            flax_name = tuple(('vit', 'embeddings', 'position_embeddings'))
            flax_params[flax_name] = param_data
        
        if name_parts[0] == "tokenizer" and name_parts[4] =="weight":
            flax_name = tuple(('vit', 'embeddings', 'patch_embeddings','projection','kernel'))
            flax_params[flax_name] = param_data
        
        if name_parts[0] == "tokenizer" and name_parts[4] =="bias":
            flax_name = tuple(('vit', 'embeddings', 'patch_embeddings','projection','bias'))
            flax_params[flax_name] = param_data
        
        if name_parts[0] == "classifier" and name_parts[1] =="fc" and name_parts[2] =="weight":
            flax_name = tuple(('classifier', 'kernel'))
            flax_params[flax_name] = jnp.transpose(param_data,(1,0))
        if name_parts[0] == "classifier" and name_parts[1] =="fc" and name_parts[2] =="bias":
            flax_name = tuple(('classifier', 'bias'))
            flax_params[flax_name] = param_data
        
        if name_parts[0] == "classifier" and name_parts[1] =="norm" and name_parts[2] =="weight":
            flax_name = tuple(('vit','layernorm' ,'scale'))
            flax_params[flax_name] = param_data
        if name_parts[0] == "classifier" and name_parts[1] =="norm" and name_parts[2] =="bias":
            flax_name = tuple(('vit','layernorm' ,'bias'))
            flax_params[flax_name] = param_data
    flax_params = flax.traverse_util.unflatten_dict(flax_params)
    flax_model.params = flax.core.unfreeze(flax.core.FrozenDict(flax_params))
    return flax_model

config = ViTConfig(
    image_size=64,  # The size of input images
    patch_size=4,  # Size of patches to be extracted from the images
    num_channels=3,  # Number of channels of the input images
    num_labels=200,  # Number of labels for classification task
    hidden_size=192,  # Dimensionality of the encoder layers and the pooler layer
    num_hidden_layers=9,  # Number of hidden layers in the Transformer encoder
    num_attention_heads=12,  # Number of attention heads for each attention layer in the Transformer encoder
    intermediate_size=384,  # Dimensionality of the "intermediate" layer in the Transformer encoder
    hidden_act="gelu",  # The non-linear activation function in the encoder and pooler
    hidden_dropout_prob=0.,  # The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler
    attention_probs_dropout_prob=0.1,  # The dropout ratio for the attention probabilities
    qkv_bias=False,
    layer_norm_eps = 1e-05
)
# Load pytorch checkpoint
checkpoint = torch.load("/home/nhd7682/MPCViT-Infer-master/mpcvit_checkpoints/mpcvit_tinyimagenet-0.1-kd.pth.tar", map_location=torch.device('cpu'))
# Create Flax model
model = CustomFlaxViTForImageClassification(config)
# Convert the pytorch weights and load them to the flax model
model = pytorch_to_flax(checkpoint['state_dict'], model)


def accuracy(predictions, labels):
    if predictions.shape != labels.shape:
        raise ValueError("Predictions and labels must have the same shape")

    correct_predictions = jnp.sum(predictions == labels)
    total_predictions = predictions.size

    return correct_predictions / total_predictions

model_test = CustomFlaxViTForImageClassification(config)
model_cipher = CustomFlaxViTForImageClassification(config=config)

def infer_cipher(inputs, params):
    outputs = model_cipher(pixel_values=inputs, params=params)
    return outputs

def run_on_spu(inputs, targets):
    # get model weights from one device
    inputs = ppd.device("P2")(lambda x: x)(inputs)
    # get input weights from another device
    params = ppd.device("P1")(lambda x: x)(model.params)
    outputs = ppd.device("SPU")(infer_cipher)(inputs, params)
    outputs = ppd.get(outputs)
    outputs = outputs['logits']
    acc = accuracy(jnp.argmax(outputs,axis=1),targets.numpy())
    print(acc)
    return acc

def run_on_cpu(inputs, targets):
    # encode context the generation is conditioned on
    # For reproducibility
    
    outputs = model_test(pixel_values=inputs, params= model.params)["logits"]
    # loss = criterion(outputs, targets)

    acc = accuracy(jnp.argmax(outputs,axis=1),targets.numpy())
    print(acc)
    return acc

if __name__ == '__main__':
    print('\n------\nRun on CPU')
    torch.manual_seed(0)
    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.480,0.448,0.397], std=[0.272,0.265,0.274]),
        transforms.RandomErasing(p=0.25),
    ]),
    'eval': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.480,0.448,0.397], std=[0.272,0.265,0.274])
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.480,0.448,0.397], std=[0.272,0.265,0.274])
    ])
    }
    dataset_eval = TinyImageNet(root='/scratch/nhd7682/data/tiny-imagenet-200/', train=False, transform=data_transforms['eval'])
    loader_eval = DataLoader(dataset=dataset_eval, batch_size=1, num_workers=1, shuffle=False)
    inputs, targets = next(iter(loader_eval))
    acc_cpu = run_on_cpu(inputs.numpy(), targets)
    print('\n------\nRun on SPU')
    acc_spu = run_on_spu(inputs.numpy(),targets)