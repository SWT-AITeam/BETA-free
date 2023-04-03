from torch import nn
from transformers import *
import torch
from torch.distributions import Normal
import torch.nn.functional as F

class RelModel(nn.Module):
    def __init__(self, config):
        super(RelModel, self).__init__()
        self.config = config
        self.bert_dim = config.bert_dim
        self.bert_encoder = BertModel.from_pretrained("bert-base-cased", cache_dir='./pre_trained_bert')
        self.relation_matrix = nn.Linear(self.bert_dim * 3, self.config.rel_num * self.config.tag_size)
        self.projection_matrix = nn.Linear(self.bert_dim * 2, self.bert_dim * 3)
        self.seq_logvar_layer = nn.Linear(self.config.rel_num * self.config.tag_size, self.config.rel_num * self.config.tag_size)

        self.dropout = nn.Dropout(self.config.dropout_prob)
        self.dropout_2 = nn.Dropout(self.config.entity_pair_dropout)
        self.activation = nn.ReLU()

    def get_encoded_text(self, token_ids, mask):
        # [batch_size, seq_len, bert_dim(768)]
        encoded_text = self.bert_encoder(token_ids, attention_mask=mask)[0]
        return encoded_text
    def _variational_layer(self, hidden, mu_layer, logvar_layer):
        sampled_z = hidden
        kld = torch.zeros(hidden.shape[:-1], dtype=torch.float32, device=hidden.device)
        if self.training and self.config.IB is True:
            mu = hidden  # 均值
            logvar = logvar_layer(hidden)  # 方差
            # TODO 训练次数设置的大点, 超5w, 8w起吧
            std = F.softplus(logvar)
            # std = torch.exp(0.5 * logvar)
            posterior = Normal(loc=mu, scale=std, validate_args=False)

            zeros = torch.zeros_like(mu, device=mu.device)
            ones = torch.ones_like(std, device=std.device)
            prior = Normal(zeros, ones, validate_args=False)

            eps = std.new_empty(std.shape)
            eps.normal_()
            sampled_z = mu + std * eps
            # (b,128)
            kld = posterior.log_prob(sampled_z).sum(-1) - prior.log_prob(sampled_z).sum(-1)

            # (b,1)
        return sampled_z, kld

    def triple_score_matrix(self, encoded_text, train = True):
        # encoded_text: [batch_size, seq_len, bert_dim(768)]
        batch_size, seq_len, bert_dim = encoded_text.size()
        # head: [batch_size, seq_len * seq_len, bert_dim(768)]
        head_representation = encoded_text.unsqueeze(2).expand(batch_size, seq_len, seq_len, bert_dim).reshape(batch_size, seq_len*seq_len, bert_dim)
        # tail: [batch_size, seq_len * seq_len, bert_dim(768)]
        tail_representation = encoded_text.repeat(1, seq_len, 1)
        # [batch_size, seq_len * seq_len, bert_dim(768)*2]
        entity_pairs = torch.cat([head_representation, tail_representation], dim=-1)

        # [batch_size, seq_len * seq_len, bert_dim(768)*3]
        entity_pairs = self.projection_matrix(entity_pairs)

        entity_pairs = self.dropout_2(entity_pairs)

        entity_pairs = self.activation(entity_pairs)

        sequence_output = self.relation_matrix(entity_pairs)
        sequence_output, kld = self._variational_layer(sequence_output, None, self.seq_logvar_layer)
        # [batch_size, seq_len * seq_len, rel_num * tag_size] -> [batch_size, seq_len, seq_len, rel_num, tag_size]
        triple_scores = sequence_output.reshape(batch_size, seq_len, seq_len, self.config.rel_num, self.config.tag_size)

        if train:
            # [batch_size, tag_size, rel_num, seq_len, seq_len]
            return triple_scores.permute(0,4,3,1,2),kld
        else:
            # [batch_size, seq_len, seq_len, rel_num]
            return triple_scores.argmax(dim = -1).permute(0,3,1,2)

    def forward(self, data, train = True):
        # [batch_size, seq_len]
        token_ids = data['token_ids']
        # [batch_size, seq_len]
        mask = data['mask']
        # [batch_size, seq_len, bert_dim(768)]
        encoded_text = self.get_encoded_text(token_ids, mask)
        encoded_text = self.dropout(encoded_text)
        # [batch_size, rel_num, seq_len, seq_len]
        output = self.triple_score_matrix(encoded_text, train)

        return output
