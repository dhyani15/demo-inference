# The flax version of mpc vit is based of hugging face implementation which is modified according to mpcvit pytorch code base. refer: https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/modeling_flax_vit.py
import json
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import flax.linen as nn
from flax.linen.dtypes import promote_dtype
from transformers.models.vit.modeling_flax_vit import FlaxViTSelfAttention, FlaxViTAttention, FlaxViTLayer, FlaxViTLayerCollection, FlaxViTEncoder, FlaxViTModule, FlaxViTForImageClassificationModule, FlaxViTPreTrainedModel, FlaxViTPooler, FlaxViTOutput, FlaxViTIntermediate, FlaxViTSelfOutput, FlaxViTForImageClassification
from transformers.models.vit.configuration_vit import ViTConfig
from transformers.modeling_flax_outputs import FlaxBaseModelOutputWithPooling, FlaxSequenceClassifierOutput, FlaxBaseModelOutput


class FlaxViTPatchEmbeddings(nn.Module):
    config: ViTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        image_size = self.config.image_size
        patch_size = self.config.patch_size
        num_patches = (image_size // patch_size) * (image_size // patch_size)
        self.num_patches = num_patches
        self.num_channels = self.config.num_channels
        self.projection = nn.Conv(
            self.config.hidden_size,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID",
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, "fan_in", "truncated_normal"
            ),
        )

    def __call__(self, pixel_values):        
        num_channels = pixel_values.shape[-1]
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        embeddings = self.projection(pixel_values)
        # print(embeddings)
        batch_size, _, _, channels = embeddings.shape
        return jnp.reshape(embeddings, (batch_size, -1, channels))

class FlaxViTEmbeddings(nn.Module):
    """Construct the CLS token, position and patch embeddings."""

    config: ViTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.cls_token = self.param(
            "cls_token",
            jax.nn.initializers.variance_scaling(self.config.initializer_range**2, "fan_in", "truncated_normal"),
            (1, 1, self.config.hidden_size),
        )
        self.patch_embeddings = FlaxViTPatchEmbeddings(self.config, dtype=self.dtype)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = self.param(
            "position_embeddings",
            jax.nn.initializers.variance_scaling(self.config.initializer_range**2, "fan_in", "truncated_normal"),
            (1, num_patches + 1, self.config.hidden_size),
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, pixel_values, deterministic=True):
        batch_size = pixel_values.shape[0]
        embeddings = self.patch_embeddings(pixel_values)
        cls_tokens = jnp.broadcast_to(self.cls_token, (batch_size, 1, self.config.hidden_size))
        embeddings = jnp.concatenate((cls_tokens, embeddings), axis=1)
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings, deterministic=deterministic)
        
        return embeddings

class CustomFlaxViTSelfAttention(FlaxViTSelfAttention):
    config: ViTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    
    def setup(self):
        self.alpha = self.param('alpha', nn.initializers.ones, (1, self.config.num_attention_heads, 1, 1))
        if self.config.hidden_size % self.config.num_attention_heads != 0:
            raise ValueError(
                "`config.hidden_size`: {self.config.hidden_size} has to be a multiple of `config.num_attention_heads`:"
                " {self.config.num_attention_heads}"
            )
        # self.qkv = nn.Dense(self.config.hidden_size * 3,dtype=self.dtype,kernel_init=jax.nn.initializers.glorot_uniform(),use_bias=False)
        self.query = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, mode="fan_in", distribution="truncated_normal"
            ),
            use_bias=self.config.qkv_bias,
        )
        self.key = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, mode="fan_in", distribution="truncated_normal"
            ),
            use_bias=self.config.qkv_bias,
        )
        self.value = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, mode="fan_in", distribution="truncated_normal"
            ),
            use_bias=self.config.qkv_bias,
        )
        
    
    def __call__(self, hidden_states, deterministic: bool = True, output_attentions: bool = False):
        head_dim = self.config.hidden_size // self.config.num_attention_heads # 1,257,192
        query_states = self.query(hidden_states).reshape( # query dense 192 -> 192 (12*16)
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )
        
        value_states = self.value(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )
        
        key_states = self.key(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )
        

        dropout_rng = None
        if not deterministic and self.config.attention_probs_dropout_prob > 0.0:
            dropout_rng = self.make_rng("dropout")


        query, key = promote_dtype(query_states, key_states, dtype=self.dtype)
        dtype = query.dtype

        assert query.ndim == key.ndim, 'q, k must have same rank.'
        assert query.shape[:-3] == key.shape[:-3], 'q, k batch dims must match.'
        assert query.shape[-2] == key.shape[-2], 'q, k num_heads must match.'
        assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

        # calculate attention matrix
        depth = query.shape[-1]
        query = query / jnp.sqrt(depth).astype(dtype)
       
        attn_weights = jnp.einsum('...qhd,...khd->...hqk', query, key, precision='highest').astype(dtype)
        
        scalattn = (attn_weights/attn_weights.shape[2]).astype(dtype) # scale attention
        attn_weights = nn.relu(attn_weights).astype(dtype) 
        attn_weights = attn_weights / (jnp.sum(attn_weights, axis=-1, keepdims=True) + 1e-5).astype(dtype) # RElusoftmax
        attn_weights = (self.alpha*attn_weights).astype(dtype) + ((1-self.alpha)*scalattn).astype(dtype) # weighted sum
        if not deterministic and self.config.attention_probs_dropout_prob > 0.0:
            keep_prob = 1.0 - self.config.attention_probs_dropout_prob
            if True:
                # dropout is broadcast across the batch + head dimensions
                dropout_shape = tuple([1] * (key.ndim - 2)) + attn_weights.shape[-2:]
                keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)  # type: ignore
            else:
                keep = random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)  # type: ignore
            multiplier = keep.astype(dtype) / jnp.asarray(keep_prob, dtype=dtype)
            attn_weights = attn_weights * multiplier

        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value_states)
        attn_output = attn_output.reshape(attn_output.shape[:2] + (-1,))

        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs



class CustomFlaxViTAttention(FlaxViTAttention):
    config: ViTConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.attention = CustomFlaxViTSelfAttention(self.config, dtype=self.dtype)
        self.output = FlaxViTSelfOutput(self.config, dtype=self.dtype)

    def __call__(self, hidden_states, deterministic=True, output_attentions: bool = False):
        return super().__call__(hidden_states, deterministic=deterministic, output_attentions=output_attentions )


class CustomFlaxViTOutput(FlaxViTOutput):
    config: ViTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, "fan_in", "truncated_normal"
            ),
            dtype=self.dtype,
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, hidden_states, attention_output, deterministic: bool = True):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = hidden_states + attention_output
        return hidden_states

class CustomFlaxViTLayer(FlaxViTLayer):
    config: ViTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.attention = CustomFlaxViTAttention(self.config, dtype=self.dtype)
        self.intermediate = FlaxViTIntermediate(self.config, dtype=self.dtype)
        self.output = CustomFlaxViTOutput(self.config, dtype=self.dtype)
        self.layernorm_before = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.layernorm_after = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    def __call__(self, hidden_states, deterministic: bool = True, output_attentions: bool = False):
        attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in ViT, layernorm is applied before self-attention
            deterministic=deterministic,
            output_attentions=output_attentions,
        )

        attention_output = attention_outputs[0]

        # first residual connection
        attention_output = attention_output + hidden_states

        # in ViT, layernorm is also applied after self-attention
        
        layer_output = self.layernorm_after(attention_output)

        hidden_states = self.intermediate(layer_output)
        
        hidden_states = self.output(hidden_states, layer_output, deterministic=deterministic)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attention_outputs[1],)
        return outputs


class CustomFlaxViTLayerCollection(FlaxViTLayerCollection):
    config: ViTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    i =0 

    def setup(self):
        self.layers = [
            CustomFlaxViTLayer(self.config, name=str(i), dtype=self.dtype) for i in range(self.config.num_hidden_layers)
        ]

    def __call__(self, hidden_states, deterministic: bool = True, output_attentions: bool = False, output_hidden_states: bool = False, return_dict: bool = True,):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer(hidden_states, deterministic=deterministic, output_attentions=output_attentions)
            

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        outputs = (hidden_states,)
        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )
        
        
        # return super().__call__(hidden_states, deterministic=deterministic, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)


class CustomFlaxViTEncoder(FlaxViTEncoder):
    config: ViTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.layer = CustomFlaxViTLayerCollection(self.config, dtype=self.dtype)

    def __call__(
        self,
        hidden_states,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        return super().__call__(hidden_states,
        deterministic = deterministic,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict)






class CustomFlaxViTModule(FlaxViTModule):
    config: ViTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    add_pooling_layer: bool = True

    def setup(self):
        self.embeddings = FlaxViTEmbeddings(self.config, dtype=self.dtype)
        self.encoder = CustomFlaxViTEncoder(self.config, dtype=self.dtype)
        self.layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.pooler = FlaxViTPooler(self.config, dtype=self.dtype) if self.add_pooling_layer else None

    def __call__(
        self,
        pixel_values,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        hidden_states = self.embeddings(pixel_values, deterministic=deterministic)

        outputs = self.encoder(
            hidden_states,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = self.layernorm(hidden_states)
        pooled = self.pooler(hidden_states) if self.add_pooling_layer else None

        if not return_dict:
            # if pooled is None, don't return it
            if pooled is None:
                return (hidden_states,) + outputs[1:]
            return (hidden_states, pooled) + outputs[1:]

        return FlaxBaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=pooled,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )






class CustomFlaxViTForImageClassificationModule(FlaxViTForImageClassificationModule):
    config: ViTConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.vit = CustomFlaxViTModule(config=self.config, dtype=self.dtype, add_pooling_layer=False)
        self.classifier = nn.Dense(
            self.config.num_labels,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, "fan_in", "truncated_normal"
            ),
        )

    def __call__(
        self,
        pixel_values=None,
        deterministic: bool = True,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        params = None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit(
            pixel_values,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.classifier(hidden_states[:, 0, :])

        if not return_dict:
            output = (logits,) + outputs[2:]
            return output

        return FlaxSequenceClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        # return super().__call__(pixel_values=pixel_values, deterministic=deterministic, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)


class CustomFlaxViTForImageClassification(FlaxViTPreTrainedModel):
    module_class = CustomFlaxViTForImageClassificationModule


