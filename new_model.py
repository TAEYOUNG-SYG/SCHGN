import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as geometric
import numpy as np
from module import Encoder, LayerNorm


def l2_loss(t):
    return torch.sum(t ** 2)


def truncated_normal_(tensor, mean=0, std=0.01):
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        return tensor


class GraphConv(torch.nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GraphConv, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv1 = geometric.nn.GCNConv(in_channel, out_channel)
        truncated_normal_(self.conv1.weight, std=np.sqrt(2.0 / (self.in_channel + self.out_channel)))
        truncated_normal_(self.conv1.bias, std=np.sqrt(2.0 / (self.in_channel + self.out_channel)))

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.tanh(x)
        return x


class MyModel(nn.Module):
    def __init__(self, data_config, args_config, device):
        super(MyModel, self).__init__()
        self.args_config = args_config
        self.data_config = data_config
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_cold = data_config['n_cold']
        self.n_health = 200
        self.n_ingredients = 33147
        self.img_size = data_config['image_size']

        self.device = device

        self.ingre_encoder = Encoder(args_config)
        self.apply(self.init_weights)

        input_channel = 64
        out_channel = 64
        self.new_gcn = GraphConv(input_channel, out_channel)

        self.emb_size = args_config.emb_size
        self.regs = eval(args_config.regs)
        self.reg_image = args_config.reg_image
        self.reg_w = args_config.reg_w
        self.reg_g = args_config.reg_g
        self.reg_health = args_config.reg_health
        self.ssl = args_config.ssl

        self.user_embed = torch.nn.Parameter(torch.FloatTensor(self.n_users, self.emb_size), requires_grad=True)
        self.item_embed = torch.nn.Parameter(torch.FloatTensor(self.n_items, self.emb_size),
                                             requires_grad=True)
        self.ingre_embed_first = torch.nn.Parameter(torch.FloatTensor(self.n_ingredients, self.emb_size),
                                                    requires_grad=True)

        self.ingre_embed_second = torch.nn.Parameter(torch.zeros(1, self.emb_size, dtype=self.ingre_embed_first.dtype),
                                                     requires_grad=False)
        self.ingre_embed_mask = torch.nn.Parameter(torch.FloatTensor(1, self.emb_size), requires_grad=True)
        self.health_embed = torch.nn.Parameter(torch.FloatTensor(self.n_health, self.emb_size), requires_grad=True)

        self.img_trans = nn.Linear(self.img_size, self.emb_size)
        truncated_normal_(self.img_trans.weight, std=np.sqrt(2.0 / (self.img_size + self.emb_size)))
        truncated_normal_(self.img_trans.bias, std=np.sqrt(2.0 / (self.img_size + self.emb_size)))

        # parameters for ingredient level attention
        self.W_att_ingre = torch.nn.Linear(self.emb_size * 3, self.emb_size)
        truncated_normal_(self.W_att_ingre.weight, std=np.sqrt(2.0 / (self.emb_size * 4)))
        truncated_normal_(self.W_att_ingre.bias, std=np.sqrt(2.0 / (self.emb_size + self.emb_size)))
        self.h_att_ingre = torch.nn.Linear(self.emb_size, 1, bias=False)
        nn.init.ones_(self.h_att_ingre.weight)

        # parameters for component level attention
        self.W_att_comp = torch.nn.Linear(self.emb_size * 2, self.emb_size)
        truncated_normal_(self.W_att_comp.weight, std=np.sqrt(2.0 / (self.emb_size * 3)))
        truncated_normal_(self.W_att_comp.bias, std=np.sqrt(2.0 / (self.emb_size + self.emb_size)))
        self.h_att_comp = torch.nn.Linear(self.emb_size, 1, bias=False)
        nn.init.ones_(self.h_att_comp.weight)

        self.W_concat = nn.Linear(self.emb_size * 3, self.emb_size)
        truncated_normal_(self.W_concat.weight, std=np.sqrt(2.0 / (self.emb_size * 4)))
        truncated_normal_(self.W_concat.bias, std=np.sqrt(2.0 / (self.emb_size + self.emb_size)))
        self.output_mlp = nn.Linear(self.emb_size, 1, bias=False)
        truncated_normal_(self.output_mlp.weight, std=np.sqrt(2.0 / (self.emb_size * 2)))

        self.mip_norm = nn.Linear(self.emb_size, self.emb_size)
        self.criterion = nn.BCELoss(reduction='none')

        self._init_weight()

    def _init_weight(self):
        truncated_normal_(self.user_embed, std=0.01)
        truncated_normal_(self.item_embed, std=0.01)
        truncated_normal_(self.ingre_embed_first, std=0.01)
        truncated_normal_(self.ingre_embed_mask, std=0.01)
        truncated_normal_(self.health_embed, std=0.01)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            # module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
            truncated_normal_(module.weight, std=0.01)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_ingre_emb(self, x, ingre_num, ingre_code):
        ingre_emb_list = []
        b_size = ingre_num.shape[0]
        for i in range(b_size):
            this_emb = x[i][: ingre_num[i]]
            ingre_emb_list.append(torch.mean(this_emb, dim=0))
        ingre_emb = torch.stack(ingre_emb_list, dim=0)
        return ingre_emb

    def sequence_mask(self, lengths, max_len):
        row = torch.arange(0, max_len, 1).to(self.device)
        matrix = torch.unsqueeze(lengths, dim=-1)
        mask = row < matrix
        return mask.type(torch.float)

    def compute_ingre_emb(self, x, ingre_num):
        max_len = 20
        mask = self.sequence_mask(ingre_num, max_len=max_len)
        ingre_emb = x * mask.unsqueeze(dim=-1)
        return ingre_emb.sum(dim=1)

    def attention_ingredient_level(self, ingre_emb, u_emb, img_emb, ingre_num):
        b = ingre_emb.shape[0]
        n = ingre_emb.shape[1]

        expand_u_emb = u_emb.unsqueeze(1)

        tile_u_emb = expand_u_emb.repeat(1, n, 1)
        expand_img = img_emb.unsqueeze(1)
        tile_img = expand_img.repeat(1, n, 1)

        concat_v = torch.cat([ingre_emb, tile_u_emb, tile_img], dim=2)

        MLP_output = torch.tanh(self.W_att_ingre(concat_v))

        A_ = self.h_att_ingre(MLP_output).squeeze()
        smooth = -1e12

        mask_mat = self.sequence_mask(ingre_num, max_len=n)
        mask_mat = torch.ones_like(mask_mat) - mask_mat
        mask_mat = mask_mat * smooth

        A = F.softmax(A_ + mask_mat, dim=1)

        A = A.unsqueeze(2)

        return torch.sum(A * ingre_emb, dim=1)

    def attention_id_ingre_image(self, u_emb, i_emb, ingre_att_emb, img_emb, hl_emb):
        b = u_emb.shape[0]

        cp1 = torch.cat([u_emb, i_emb], dim=1)
        cp2 = torch.cat([u_emb, ingre_att_emb], dim=1)
        cp3 = torch.cat([u_emb, img_emb], dim=1)
        cp4 = torch.cat([u_emb, hl_emb], dim=1)

        cp = torch.cat([cp1, cp2, cp3, cp4], dim=0)

        c_hidden_output = torch.tanh(self.W_att_comp(cp))

        c_mlp_output = self.h_att_comp(c_hidden_output).view(b, -1)

        B = F.softmax(c_mlp_output, dim=1).unsqueeze(2)
        ce1 = i_emb.unsqueeze(1)  # [b, 1, e]
        ce2 = ingre_att_emb.unsqueeze(1)  # [b, 1, e]
        ce3 = img_emb.unsqueeze(1)  # [b, 1, e]
        ce4 = hl_emb.unsqueeze(1)   # [b, 1, e]
        ce = torch.cat([ce1, ce2, ce3, ce4], dim=1)  # [b, 4, e]
        return torch.sum(B * ce, dim=1)  # [b, e]

    def masked_ingre_prediction(self, ingre_emb, target_emb):
        ingre_emb = self.mip_norm(ingre_emb.view([-1, self.emb_size]))
        target_emb = target_emb.view([-1, self.emb_size])
        score = torch.mul(ingre_emb, target_emb)
        return torch.sigmoid(torch.sum(score, -1))

    def compute_ssl_loss(self, ingre_embedding, ingre_embedding_gcn, masked_ingre_seq, pos_ingre, neg_ingre):

        ingre_emb = ingre_embedding_gcn[masked_ingre_seq]

        seq_mask = (masked_ingre_seq == self.n_ingredients).float() * -1e8
        seq_mask = torch.unsqueeze(torch.unsqueeze(seq_mask, 1), 1)
        encoded_embs = self.ingre_encoder(ingre_emb, seq_mask, output_all_encoded_layers=True)
        new_ingre_emb = encoded_embs[-1]

        pos_ingre_emb = ingre_embedding[pos_ingre]
        neg_ingre_emb = ingre_embedding[neg_ingre]
        pos_score = self.masked_ingre_prediction(new_ingre_emb, pos_ingre_emb)
        neg_score = self.masked_ingre_prediction(new_ingre_emb, neg_ingre_emb)
        mip_distance = torch.sigmoid(pos_score - neg_score)
        mip_loss = self.criterion(mip_distance, torch.ones_like(mip_distance, dtype=torch.float32))
        mip_mask = (masked_ingre_seq == self.n_ingredients+1).float()
        num_tokens = torch.sum(mip_mask)
        mip_loss = torch.sum(mip_loss * mip_mask.flatten())
        return mip_loss

    def compute_score(self, user, item, ingre, ingre_num, img, hl, is_training, g2i_edges, i2u_edges, ingre_embedding):

        u_emb = self.user_embed[user]
        i_emb = self.item_embed[item]
        ingre_emb = ingre_embedding[ingre]
        hl_emb = self.health_embed[hl]
        img_emb = self.img_trans(img)
        edge_index = torch.cat([g2i_edges, i2u_edges], dim=0)

        x = torch.cat([
            self.user_embed, self.item_embed,
            self.ingre_embed_first, self.health_embed
        ], dim=0)
        gcn_emb = self.new_gcn(x, edge_index.t().contiguous())  # [n_nodes, emb_size]
        user_embed_gcn, item_embed_gcn, ingre_embed_gcn, hl_emb_gcn = torch.split(
            gcn_emb, [self.n_users, self.n_items, self.n_ingredients, self.n_health], dim=0
        )
        ingre_embedding_gcn = torch.cat([ingre_embed_gcn, self.ingre_embed_second, self.ingre_embed_mask], dim=0)

        u_gcn_emb = user_embed_gcn[user]
        i_gcn_emb = item_embed_gcn[item]
        ingre_gcn_emb = ingre_embedding_gcn[ingre]
        hl_gcn_emb = hl_emb_gcn[hl]

        u_emb_final = u_emb + u_gcn_emb
        i_emb_final = i_emb + i_gcn_emb
        ingre_emb_final = ingre_emb + ingre_gcn_emb
        hl_emb_final = hl_emb + hl_gcn_emb

        ingre_att_emb = self.attention_ingredient_level(ingre_emb_final, u_emb_final, img_emb, ingre_num)
        item_att_emb = self.attention_id_ingre_image(u_emb_final, i_emb_final, ingre_att_emb, img_emb, hl_emb_final)
        ui_concat_emb = torch.cat([u_emb_final, item_att_emb, u_emb_final * item_att_emb], dim=1)
        hidden_input = self.W_concat(ui_concat_emb)
        MLP_ouput = F.relu(F.dropout(hidden_input, p=0.5, training=is_training))
        score = self.output_mlp(MLP_ouput).squeeze()

        return score, u_emb, i_emb, ingre_emb, hl_emb, ingre_embedding_gcn, item_att_emb

    def forward(self, user,
                pos_item, pos_ingre, pos_ingre_num, pos_img, pos_hl,
                neg_item, neg_ingre, neg_ingre_num, neg_img, neg_hl,
                g2i_edges, i2u_edges,
                masked_ingre_seq, pos_ingre_seq, neg_ingre_seq
                ):
        ingre_embedding = torch.cat([self.ingre_embed_first, self.ingre_embed_second, self.ingre_embed_mask], dim=0)
        pos_scores, user_emb, pos_item_emb, pos_ingre_emb, pos_hl_emb, ingre_embedding_g, _ = self.compute_score(user, pos_item, pos_ingre, pos_ingre_num,
                                                                               pos_img, pos_hl, True, g2i_edges, i2u_edges, ingre_embedding)
        neg_scores, user_emb, neg_item_emb, neg_ingre_emb, neg_hl_emb, _, _ = self.compute_score(user, neg_item, neg_ingre, neg_ingre_num,
                                                                               neg_img, neg_hl, True, g2i_edges, i2u_edges, ingre_embedding)

        ssl_loss = self.ssl * self.compute_ssl_loss(ingre_embedding, ingre_embedding_g, masked_ingre_seq, pos_ingre_seq, neg_ingre_seq)

        bpr_loss = torch.log(torch.sigmoid(pos_scores - neg_scores))
        bpr_loss = -torch.sum(bpr_loss)

        reg_loss = self.regs * (l2_loss(user_emb) + l2_loss(pos_item_emb) + l2_loss(neg_item_emb) + l2_loss(pos_ingre_emb) + l2_loss(neg_ingre_emb))
        reg_loss += self.reg_health * (l2_loss(pos_hl_emb) + l2_loss(neg_hl_emb))
        reg_loss += self.reg_image * (l2_loss(self.img_trans.weight))
        reg_loss += self.reg_w * (l2_loss(self.W_concat.weight) + l2_loss(
            self.output_mlp.weight))
        reg_loss += self.reg_g * (l2_loss(self.new_gcn.conv1.weight))

        loss = bpr_loss + reg_loss + ssl_loss
        return loss, bpr_loss, reg_loss, ssl_loss

    def inference(self, user, item, ingre, ingre_num, img, hl, g2i_edges, i2u_edges):
        ingre_embedding = torch.cat([self.ingre_embed_first, self.ingre_embed_second], dim=0)
        predictions, user_emb, item_emb, ingre_emb, hl_emb, _, r_final_emb = self.compute_score(user, item, ingre, ingre_num, img, hl, False, g2i_edges, i2u_edges, ingre_embedding)

        return predictions, r_final_emb
