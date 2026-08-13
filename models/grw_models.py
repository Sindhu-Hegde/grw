import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math, copy
import random

from models.modules import *
from transformers import AutoTokenizer, XLMRobertaModel

from einops.layers.torch import Rearrange
from einops import rearrange

from transformers import ResNetConfig, ResNetModel


import torch
import torch.nn as nn


class Semantic_Classifier(nn.Module):

	def __init__(self, input_dim=768, num_classes=1, N_encoder=3, N_decoder=3, d_model=512, d_ff=2048, h=8, dropout=0.1, use_layer_weights=True):
		super().__init__()

		self.use_layer_weights = use_layer_weights
		self.num_classes = num_classes
		self.d_model = d_model

		# SHuBERT layer aggregation  
		if self.use_layer_weights:
			self.layer_weights = nn.Sequential(
									nn.Linear(input_dim, 128),
									nn.ReLU(inplace=True),
									nn.Linear(128, 1)
								)
		# Input projection
		self.input_proj = nn.Sequential(nn.Linear(input_dim, d_model), 
									nn.LayerNorm(d_model),
									nn.ReLU(), 
									nn.Linear(d_model, d_model),)

		# Encoder for full long context
		c = copy.deepcopy
		attn = MultiHeadedAttention_Transformer_ROPE(h, d_model, dropout=dropout)
		ff = PositionwiseFeedForward_Transformer(d_model, d_ff, dropout)
		self.gesture_encoder = Encoder_Transformer(EncoderLayer_Transformer(d_model, c(attn), c(ff), dropout), N_encoder)

		# Transformer Decoder for cross-attention
		decoder_layer = nn.TransformerDecoderLayer(
			d_model=d_model, 
			nhead=h, 
			dim_feedforward=d_ff,
			dropout=dropout, 
			activation='relu', 
			batch_first=True
		)
		self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=N_decoder)

		# Classifier projection
		self.gesture_proj = nn.Sequential(nn.Linear(d_model, d_model), 
									nn.LayerNorm(d_model),
									nn.ReLU(),
									nn.Linear(d_model, num_classes))

		

	def forward(self, x, target_intervals, x_mask=None):

		# print("X: ", x.shape)

		if self.use_layer_weights:
			x = x.permute(0, 2, 1, 3)  # (B, L, T, 768)
			scores = self.layer_weights(x).squeeze(-1)  	# (B, L, T)
			alpha = F.softmax(scores, dim=1).unsqueeze(-1)  # (B, L, T, 1)
			x = (x * alpha).sum(dim=1)  					# (B, T, 768)

		# Project and encode full long context sequence
		x = self.input_proj(x)
		# print("Input proj: ", x.shape)                                            # BxTx512

		# Encode full sequence (long context ~250 frames)
		gesture_emb = self.gesture_encoder(x, x_mask)  # B x T x D
		# print("Encoded full context: ", gesture_emb.shape)     				    # BxTx512
		
		# Extract target intervals for each sample (these will be queries)
		B = gesture_emb.shape[0]
		device = gesture_emb.device
		target_intervals = target_intervals.to(device)

		query_list = []
		query_lengths = []
		for i in range(B):
			start = int(target_intervals[i, 0])
			end = int(target_intervals[i, 1]) + 1  # +1 to make end inclusive
			query = gesture_emb[i, start:end, :]  # (interval_length, D)
			query_list.append(query)
			query_lengths.append(query.shape[0])
		
		# Pad queries to same length for batching
		max_query_len = max(query_lengths)
		query_padded = torch.zeros(B, max_query_len, self.d_model, device=device)
		query_mask = torch.zeros(B, max_query_len, dtype=torch.bool, device=device)
		
		for i, query in enumerate(query_list):
			query_len = query_lengths[i]
			query_padded[i, :query_len, :] = query
			query_mask[i, :query_len] = True
		
		# print("Query (target interval): ", query_padded.shape)  # B x max_query_len x D
		# print("Query mask: ", query_mask.shape)  # B x max_query_len
		
		# Transformer Decoder: query (target interval) cross-attends to memory (full context)
		# memory: full encoded sequence (B x T x D)
		# tgt: target interval query (B x max_query_len x D)
		# memory_key_padding_mask: mask for memory (B x T), 1 for padding
		memory_key_padding_mask = None
		if x_mask is not None:
			# x_mask is B x 1 x T, convert to B x T and invert (1 for padding, 0 for valid)
			if x_mask.dim() == 3:
				memory_key_padding_mask = ~x_mask.squeeze(1).bool()  # B x T
			else:
				memory_key_padding_mask = ~x_mask.bool()  # B x T
		# print("Memory key padding mask: ", memory_key_padding_mask.shape)

		# Decoder cross-attention: query attends to full context
		decoded = self.decoder(
			tgt=query_padded, 
			memory=gesture_emb,
			tgt_key_padding_mask=~query_mask,  # True for padding positions
			memory_key_padding_mask=memory_key_padding_mask
		)  # B x max_query_len x D
		# print("Decoded output: ", decoded.shape)  # B x max_query_len x D
		
		# Pool decoded output: take mean over actual query length (not padding)
		decoded_pooled = []
		for i in range(B):
			query_len = query_lengths[i]
			decoded_pooled.append(torch.mean(decoded[i, :query_len, :], dim=0))  # (D,)
		decoded_pooled = torch.stack(decoded_pooled, dim=0)  # (B, D)
		# print("Decoded pooled: ", decoded_pooled.shape)  # B x D
		
		# Classification
		output_emb = self.gesture_proj(decoded_pooled)
		# print("Output MLP: ", output_emb.shape)						            # Bxnum_classes
		
		return output_emb, gesture_emb

class Word_Recognition_Localization(nn.Module):

	def __init__(self, input_dim=768, num_classes=100, N=6, d_model=768, d_ff=2048, h=12, dropout=0.1, use_layer_weights=True):
		super().__init__()

		self.use_layer_weights = use_layer_weights
		self.num_classes = num_classes

		# SHuBERT layer aggregation 
		self.layer_weights = nn.Sequential(
								nn.Linear(input_dim, 128),
								nn.ReLU(inplace=True),
								nn.Linear(128, 1)
							)

		# Input projection
		self.input_proj = nn.Sequential(nn.Linear(input_dim, d_model), 
									nn.LayerNorm(d_model),
									nn.ReLU(), 
									nn.Linear(d_model, d_model),)

		# Transformer Encoder
		c = copy.deepcopy
		attn = MultiHeadedAttention_Transformer_ROPE(h, d_model, dropout=dropout)
		ff = PositionwiseFeedForward_Transformer(d_model, d_ff, dropout)
		self.position_enc = PositionalEncoding_Transformer(d_model, dropout)
		self.gesture_encoder = Encoder_Transformer(EncoderLayer_Transformer(d_model, c(attn), c(ff), dropout), N)

		# Gesture Recognition Head
		self.cls_head = nn.Sequential(nn.Linear(d_model, d_model), 
									nn.LayerNorm(d_model),
									nn.ReLU(),
									nn.Linear(d_model, num_classes))

		# Gesture Localization Head
		self.loc_head = nn.Sequential(
			nn.Linear(d_model, d_model//2),
			nn.ReLU(inplace=True),
			nn.Linear(d_model//2, 1)
		)  


	def forward(self, x, x_mask=None):

		# print("Input shape: ", x.shape)

		if self.use_layer_weights:
			x = x.permute(0, 2, 1, 3)  # (B, L, T, 768)
			scores = self.layer_weights(x).squeeze(-1)  	# (B, L, T)
			alpha = F.softmax(scores, dim=1).unsqueeze(-1)  # (B, L, T, 1)
			x = (x * alpha).sum(dim=1)  					# (B, T, 768)


		x = self.input_proj(x)
		# print("Input proj: ", x.shape)                                       	# BxTx768

		position_emb = self.position_enc(x)
		# print("Position encoding: ", position_emb.shape)						# BxTx768
	
		# print("Input mask: ", x_mask.shape)								 	# Bx1xT

		per_frame_emb = self.gesture_encoder(position_emb, x_mask)
		# print("Transformer: ", per_frame_emb.shape)     				  		# BxTx768

		B, T, D = per_frame_emb.shape
		flat = per_frame_emb.view(B * T, D)
		loc_logits = self.loc_head(flat).view(B, T)
		# print("Localization logits: ", loc_logits.shape)						# BxT

		# Attention-weighted pooling: classification is guided by loc_logits
		# so the classification loss backprops into loc_logits and trains it
		# to attend to the temporally discriminative (gesture) frames.
		loc_weights = torch.sigmoid(loc_logits)  # (B, T)
		if x_mask is not None:
			mask_sq = x_mask.squeeze(1) if x_mask.dim() == 3 else x_mask
			loc_weights = loc_weights * mask_sq.float()
		denom = loc_weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
		weighted_emb = (per_frame_emb * loc_weights.unsqueeze(-1)).sum(dim=1) / denom
		# print("Pooled embeddings: ", weighted_emb.shape)						# Bx768

		cls_logits = self.cls_head(weighted_emb)
		# print("Recognition logits: ", cls_logits.shape)						# Bxnum_class

		return {
			'cls_logits': cls_logits,
			'loc_logits': loc_logits,
			'per_frame_emb': per_frame_emb,
		}
		