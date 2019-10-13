from torch import nn
import torch

class _MultiLayerPercep(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(_MultiLayerPercep, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2, bias=True),
            nn.ReLU(),            
            nn.Linear(input_dim // 2, output_dim, bias=True),
        )

    def forward(self, x):
        return self.mlp(x)


class _Aggregation(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(_Aggregation, self).__init__()
        self.aggre = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=True),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.aggre(x)


class _UserModel(nn.Module):
    ''' User modeling to learn user latent factors.
    User modeling leverages two types aggregation: item aggregation and social aggregation
    '''
    def __init__(self, emb_dim, user_emb, item_emb, rate_emb):
        super(_UserModel, self).__init__()
        self.user_emb = user_emb
        self.item_emb = item_emb
        self.rate_emb = rate_emb
        self.emb_dim = emb_dim

        self.g_v = _MultiLayerPercep(2 * self.emb_dim, self.emb_dim)

        self.user_items_att = _MultiLayerPercep(2 * self.emb_dim, 1)
        self.aggre_items = _Aggregation(self.emb_dim, self.emb_dim)

        self.user_users_att = _MultiLayerPercep(2 * self.emb_dim, 1)
        self.aggre_neigbors = _Aggregation(self.emb_dim, self.emb_dim)
        
        self.combine_mlp = nn.Sequential(
            nn.Linear(2 * self.emb_dim, self.emb_dim, bias = True),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim, bias = True),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim, bias = True),
            nn.ReLU(),
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # used for preventing zero div error when calculating softmax score
        self.eps = 1e-10

    def forward(self, uids, u_item_pad, u_user_pad, u_user_item_pad):
        # item aggregation
        q_a = self.item_emb(u_item_pad[:,:,0])   # B x maxi_len x emb_dim
        mask_u = torch.where(u_item_pad[:,:,0] > 0, torch.tensor([1.], device=self.device), torch.tensor([0.], device=self.device))   # B x maxi_len
        u_item_er = self.rate_emb(u_item_pad[:,:,1])  # B x maxi_len x emb_dim
        
        x_ia = self.g_v(torch.cat([q_a, u_item_er], dim = 2).view(-1, 2 * self.emb_dim)).view(q_a.size())  # B x maxi_len x emb_dim

        ## calculate attention scores in item aggregation 
        p_i = mask_u.unsqueeze(2).expand_as(x_ia) * self.user_emb(uids).unsqueeze(1).expand_as(x_ia)  # B x maxi_len x emb_dim
        
        alpha = self.user_items_att(torch.cat([x_ia, p_i], dim = 2).view(-1, 2 * self.emb_dim)).view(mask_u.size()) # B x maxi_len
        alpha = torch.exp(alpha) * mask_u
        alpha = alpha / (torch.sum(alpha, 1).unsqueeze(1).expand_as(alpha) + self.eps)

        h_iI = self.aggre_items(torch.sum(alpha.unsqueeze(2).expand_as(x_ia) * x_ia, 1))     # B x emb_dim

        # social aggregation
        q_a_s = self.item_emb(u_user_item_pad[:,:,:,0])   # B x maxu_len x maxi_len x emb_dim
        mask_s = torch.where(u_user_item_pad[:,:,:,0] > 0, torch.tensor([1.], device=self.device), torch.tensor([0.], device=self.device))  # B x maxu_len x maxi_len
        u_user_item_er = self.rate_emb(u_user_item_pad[:,:,:,1]) # B x maxu_len x maxi_len x emb_dim
        
        x_ia_s = self.g_v(torch.cat([q_a_s, u_user_item_er], dim = 3).view(-1, 2 * self.emb_dim)).view(q_a_s.size())  # B x maxu_len x maxi_len x emb_dim   

        p_i_s = mask_s.unsqueeze(3).expand_as(x_ia_s) * self.user_emb(u_user_pad).unsqueeze(2).expand_as(x_ia_s)  # B x maxu_len x maxi_len x emb_dim

        alpha_s = self.user_items_att(torch.cat([x_ia_s, p_i_s], dim = 3).view(-1, 2 * self.emb_dim)).view(mask_s.size())    # B x maxu_len x maxi_len
        alpha_s = torch.exp(alpha_s) * mask_s
        alpha_s = alpha_s / (torch.sum(alpha_s, 2).unsqueeze(2).expand_as(alpha_s) + self.eps)

        h_oI_temp = torch.sum(alpha_s.unsqueeze(3).expand_as(x_ia_s) * x_ia_s, 2)    # B x maxu_len x emb_dim
        h_oI = self.aggre_items(h_oI_temp.view(-1, self.emb_dim)).view(h_oI_temp.size())  # B x maxu_len x emb_dim

        ## calculate attention scores in social aggregation
        beta = self.user_users_att(torch.cat([h_oI, self.user_emb(u_user_pad)], dim = 2).view(-1, 2 * self.emb_dim)).view(u_user_pad.size())
        mask_su = torch.where(u_user_pad > 0, torch.tensor([1.], device=self.device), torch.tensor([0.], device=self.device))
        beta = torch.exp(beta) * mask_su
        beta = beta / (torch.sum(beta, 1).unsqueeze(1).expand_as(beta) + self.eps)
        h_iS = self.aggre_neigbors(torch.sum(beta.unsqueeze(2).expand_as(h_oI) * h_oI, 1))     # B x emb_dim

        ## learning user latent factor
        h_i = self.combine_mlp(torch.cat([h_iI, h_iS], dim = 1))

        return h_i


class _ItemModel(nn.Module):
    '''Item modeling to learn item latent factors.
    '''
    def __init__(self, emb_dim, user_emb, item_emb, rate_emb):
        super(_ItemModel, self).__init__()
        self.emb_dim = emb_dim
        self.user_emb = user_emb
        self.item_emb = item_emb
        self.rate_emb = rate_emb

        self.g_u = _MultiLayerPercep(2 * self.emb_dim, self.emb_dim)
        
        self.item_users_att = _MultiLayerPercep(2 * self.emb_dim, 1)
        self.aggre_users = _Aggregation(self.emb_dim, self.emb_dim)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # used for preventing zero div error when calculating softmax score
        self.eps = 1e-10

    def forward(self, iids, i_user_pad):
        # user aggregation
        p_t = self.user_emb(i_user_pad[:,:,0])
        mask_i = torch.where(i_user_pad[:,:,0] > 0, torch.tensor([1.], device=self.device), torch.tensor([0.], device=self.device))
        i_user_er = self.rate_emb(i_user_pad[:,:,1])
        
        f_jt = self.g_u(torch.cat([p_t, i_user_er], dim = 2).view(-1, 2 * self.emb_dim)).view(p_t.size())
        
        # calculate attention scores in user aggregation
        q_j = mask_i.unsqueeze(2).expand_as(f_jt) * self.item_emb(iids).unsqueeze(1).expand_as(f_jt)
        
        miu = self.item_users_att(torch.cat([f_jt, q_j], dim = 2).view(-1, 2 * self.emb_dim)).view(mask_i.size())
        miu = torch.exp(miu) * mask_i
        miu = miu / (torch.sum(miu, 1).unsqueeze(1).expand_as(miu) + self.eps)
        
        z_j = self.aggre_users(torch.sum(miu.unsqueeze(2).expand_as(f_jt) * f_jt, 1))

        return z_j


class GraphRec(nn.Module):
    '''GraphRec model proposed in the paper Graph neural network for social recommendation 

    Args:
        number_users: the number of users in the dataset.
        number_items: the number of items in the dataset.
        num_rate_levels: the number of rate levels in the dataset.
        emb_dim: the dimension of user and item embedding (default = 64).

    '''
    def __init__(self, num_users, num_items, num_rate_levels, emb_dim = 64):
        super(GraphRec, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_rate_levels = num_rate_levels
        self.emb_dim = emb_dim
        self.user_emb = nn.Embedding(self.num_users, self.emb_dim, padding_idx = 0)
        self.item_emb = nn.Embedding(self.num_items, self.emb_dim, padding_idx = 0)
        self.rate_emb = nn.Embedding(self.num_rate_levels, self.emb_dim, padding_idx = 0)

        self.user_model = _UserModel(self.emb_dim, self.user_emb, self.item_emb, self.rate_emb)

        self.item_model = _ItemModel(self.emb_dim, self.user_emb, self.item_emb, self.rate_emb)
        
        self.rate_pred = nn.Sequential(
            nn.Linear(2 * self.emb_dim, self.emb_dim, bias = True),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim, bias = True),
            nn.ReLU(),
            nn.Linear(self.emb_dim, 1),
        )


    def forward(self, uids, iids, u_item_pad, u_user_pad, u_user_item_pad, i_user_pad):
        '''
        Args:
            uids: the user id sequences.
            iids: the item id sequences.
            u_item_pad: the padded user-item graph.
            u_user_pad: the padded user-user graph.
            u_user_item_pad: the padded user-user-item graph.
            i_user_pad: the padded item-user graph.

        Shapes:
            uids: (B).
            iids: (B).
            u_item_pad: (B, ItemSeqMaxLen, 2).
            u_user_pad: (B, UserSeqMaxLen).
            u_user_item_pad: (B, UserSeqMaxLen, ItemSeqMaxLen, 2).
            i_user_pad: (B, UserSeqMaxLen, 2).

        Returns:
            the predicted rate scores of the user to the item.
        '''

        h_i = self.user_model(uids, u_item_pad, u_user_pad, u_user_item_pad)
        z_j = self.item_model(iids, i_user_pad)

        # make prediction
        r_ij = self.rate_pred(torch.cat([h_i, z_j], dim = 1))

        return r_ij
