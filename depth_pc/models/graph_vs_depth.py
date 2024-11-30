import os

os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn.norm as norm
from torch_geometric.data import Batch
from torch_geometric.nn import (AttentionalAggregation, MessagePassing,
                                PointTransformerConv)

from ..midend.graph_gen_depth import GraphData
from .functions import objectives, postprocess


class MLP(nn.Module):
    def __init__(self, channels, bias=True):
        super().__init__()
        assert len(channels) >= 2
        fcs = []
        for i in range(1, len(channels)):
            fcs.append(nn.Linear(channels[i-1], channels[i], bias=bias))
        self.fcs = nn.ModuleList(fcs)
    
    def forward(self, x):
        fc = self.fcs[0]
        x = fc(x)
        for i in range(1, len(self.fcs)):
            x = F.relu(x)
            x = self.fcs[i](x)
        return x


# $\mathbf{x}_{i}^{(k)}=\gamma^{(k)} (\mathbf{x} _{i}^{(k-1)}, 
#                                     \square _{j \in \mathcal{N}(i)} \phi^{(k)}(\mathbf{x} _{i}^{(k-1)}, 
#                                                                                \mathbf{x} _{j}^{(k-1)}, 
#                                                                                \mathbf{e} _{i, j}))$
#
# x = ...           # Node features of shape [num_nodes, num_features]
# edge_index = ...  # Edge indices of shape [2, num_edges]

# x_j = x[edge_index[0]]  # Source node features [num_edges, num_features]
# x_i = x[edge_index[1]]  # Target node features [num_edges, num_features]


class PointEdgeConv(MessagePassing):
    """
    Combination of PointConv, EdgeConv and GraphConv. 
    Codes are adapted from torch_geometric.

    Refs:
        - PointConv: "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation",
            <https://arxiv.org/abs/1612.00593>
        - EdgeConv: "Dynamic Graph CNN for Learning on Point Clouds", 
            <https://arxiv.org/abs/1801.07829>

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        msg_nn: mapping [-1, in_channels[1]*2 + pos_channels] to [-1, out_channels].
        att_nn: mapping [-1, out_channels + pos_channels] to [-1, out_channels]
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        msg_nn: nn.Module, 
        **kwargs
    ):
        super(PointEdgeConv, self).__init__(aggr="max", **kwargs)

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.msg_nn = msg_nn
        if in_channels[0] == in_channels[1]:
            self.lin_x_src = nn.Identity()
        else:
            self.lin_x_src = nn.Linear(in_channels[0], in_channels[1])

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self.msg_nn, "reset_parameters"):
            self.msg_nn.reset_parameters()
        if isinstance(self.lin_x_src, nn.Linear):
            self.lin_x_src.reset_parameters()

    def forward(self, x, pos, edge_index):
        if isinstance(x, torch.Tensor):
            x = (x, x)
        else:
            x = (self.lin_x_src(x[0]), x[1])
        
        if isinstance(pos, torch.Tensor):
            pos = (pos, pos)
        
        out = self.propagate(edge_index, x=x, pos=pos, size=None)
        return out

    def message(self, x_j, x_i, pos_j, pos_i):
        msg = torch.cat([x_i, x_j - x_i, pos_j - pos_i], dim=-1)
        msg = self.msg_nn(msg)  # (..., out_channels)

        return msg


class PERConv(nn.Module):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        pos_channels: int,
    ):
        super(PERConv, self).__init__()

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pos_channels = pos_channels

        msg_nn = MLP([in_channels[1]*2 + pos_channels, out_channels])
        self.conv = PointEdgeConv(in_channels, out_channels, msg_nn)
        self.bn = norm.GraphNorm(out_channels)
        self.fc = nn.Linear(out_channels, out_channels)

        if in_channels[1] != out_channels:
            self.skip = nn.Linear(in_channels[1], out_channels, bias=False)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x, pos, edge_index, batch):
        out = self.conv(x, pos, edge_index)
        out = self.bn(out, batch)
        out = F.relu(out)

        out = self.fc(out)
        if isinstance(x, torch.Tensor):
            out += self.skip(x)
        else:
            if x[1] is not None:
                out += self.skip(x[1])
        
        return out


class PEConvGRUCell(nn.Module):
    def __init__(self, in_channels, out_channels, pos_channels):
        super(PEConvGRUCell, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pos_channels = pos_channels

        self.conv_gates = PointEdgeConv(
            in_channels=in_channels+out_channels,
            out_channels=out_channels*2,
            msg_nn=MLP([(in_channels+out_channels)*2 + pos_channels, out_channels*2])
        )
        self.conv_candi = PointEdgeConv(
            in_channels=in_channels+out_channels,
            out_channels=out_channels,
            msg_nn=MLP([(in_channels+out_channels)*2 + pos_channels, out_channels])
        )
    
    def init_hidden(self, size0):
        zeros = torch.zeros(size0, self.out_channels)
        return zeros
    
    def forward(self, h, x, pos, edge_index_gate, edge_index_cand):
        combined = torch.cat([x, h], dim=-1)
        gates = torch.sigmoid(self.conv_gates(combined, pos, edge_index_gate))
        reset_gate, update_gate = torch.chunk(gates, chunks=2, dim=-1)

        combined = torch.cat([x, h*reset_gate], dim=-1)
        ht = torch.tanh(self.conv_candi(combined, pos, edge_index_cand))

        h_next = (1 - update_gate) * h + update_gate * ht
        return h_next


class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, pos_dim):
        super(Encoder, self).__init__()
        
        self.pathway1 = nn.Linear(2, output_dim)
        self.pathway2 = nn.Linear(1, output_dim)
        self.depth_embedding = MLP([output_dim *2, output_dim*2, 6])
        self.depth_embedding_norm = MLP([output_dim *2, output_dim*2, 1])
        self.att_clu_conv = PointTransformerConv(
            in_channels=output_dim * 2 , # 128
            out_channels=output_dim * 2, # 128
            pos_nn=MLP([pos_dim*2, output_dim, output_dim*2]), # [4, 128, 128]
            attn_nn=MLP([output_dim*2, output_dim*2, output_dim*2]), # [2, 128, 128]
            add_self_loops=False
        )
        self.out_enc = MLP([output_dim*2 *2, output_dim*2, output_dim])

    def get_clu(self, x, x_ref, pos, pos_ref, 
                l0_to_l1_edge_index, centers_index):
        
        x = (x, x_ref[centers_index])                           # ([n_points,128], [n_clusters,128])
        pos_ref_center = pos_ref[centers_index]                 # [n_clusters,2]
        pos = (
            torch.cat([pos, pos_ref], dim=-1),                  # ([n_points,4],
            torch.cat([pos_ref_center, pos_ref_center], dim=-1) #  [n_clusters,4])
        )       
        x_clu = F.relu(self.att_clu_conv(x, pos, l0_to_l1_edge_index)) # [2, n_clusters*n_clusters]
        return x_clu

    def ca_fusion(self, x, depth):
        x = self.pathway1(x)                  # [num_points, 128]
        depth = self.pathway2(depth[:, None]) # [num_points, 128]

        num_points = x.shape[0]
        BLOCK_SIZE = 1024
        num_blocks = (num_points + BLOCK_SIZE - 1) // BLOCK_SIZE

        fused_x = torch.zeros_like(x)
        fused_depth = torch.zeros_like(depth)

        for i in range(num_blocks):
            start = i * BLOCK_SIZE
            end = min((i+1) * BLOCK_SIZE, num_points)
            
            block_x = x[start:end]
            block_depth = depth[start:end]
            
            scores = torch.matmul(block_x, depth.transpose(0, 1))[:end-start, :end-start]
            attentions_a = torch.softmax(scores, dim=-1)
            attentions_b = torch.softmax(scores.transpose(0, 1), dim=-1)
            
            block_x_updated = torch.matmul(attentions_b, block_depth)[:end-start]
            block_depth_updated = torch.matmul(attentions_a, block_x)[:end-start]
            
            fused_x[start:end] = block_x_updated
            fused_depth[start:end] = block_depth_updated

        return fused_x, fused_depth

    def forward(self, x_cur, x_tar, pos_cur, pos_tar, cluster_mask, 
                l0_to_l1_edge_index, centers_index, depth_cur, depth_tar):
        
        x_cur, depth_cur = self.ca_fusion(x_cur, depth_cur) 
        x_tar, depth_tar = self.ca_fusion(x_tar, depth_tar)
        x_cur = torch.cat([x_cur, depth_cur], dim=-1)
        x_tar = torch.cat([x_tar, depth_tar], dim=-1) 
        
        dist_embedding = self.depth_embedding(torch.cat([depth_cur, depth_tar - depth_cur], dim=-1))
        dist_embedding_norm = self.depth_embedding_norm(torch.cat([depth_cur, depth_tar - depth_cur], dim=-1))

        x_cur = F.relu(x_cur)
        x_tar = F.relu(x_tar)

        x_cur_clu = self.get_clu(
            x_cur, x_tar, pos_cur, pos_tar, 
            l0_to_l1_edge_index, centers_index
        )
        x_tar_clu = self.get_clu(
            x_tar, x_tar, pos_tar, pos_tar, 
            l0_to_l1_edge_index, centers_index
        )
        x_clu = torch.cat([x_cur_clu, x_tar_clu - x_cur_clu], dim=-1)
        x_clu = F.relu(self.out_enc(x_clu)) * cluster_mask.float().unsqueeze(-1)
        return x_clu, dist_embedding, dist_embedding_norm


class Backbone(nn.Module):
    def __init__(self, hidden_dim, pos_dim):
        super(Backbone, self).__init__()

        self.l1_conv0 = PERConv(hidden_dim, hidden_dim, pos_dim)
        self.temporal_aggr = PEConvGRUCell(hidden_dim, hidden_dim, pos_dim)
        self.l1_conv1 = PERConv(hidden_dim, hidden_dim, pos_dim)
    
    def forward(self, hidden, x_clu, pos_clu, 
                l1_dense_edge_index_cur, l1_dense_edge_index_tar, batch_clu):
        x_clu = F.relu(self.l1_conv0(x_clu, pos_clu, l1_dense_edge_index_cur, batch_clu))
        hidden = self.temporal_aggr(
            hidden, x_clu, pos_clu, 
            edge_index_gate=l1_dense_edge_index_tar,
            edge_index_cand=l1_dense_edge_index_cur
        )
        x_clu = F.relu(self.l1_conv1(hidden, pos_clu, l1_dense_edge_index_cur, batch_clu))
        return hidden, x_clu


class Decoder(nn.Module):
    def __init__(self, input_dim, regress_norm=False):
        super(Decoder, self).__init__()

        self.regress_norm = regress_norm
        self.pool = AttentionalAggregation(
            gate_nn=nn.Linear(input_dim, 1)
        )
        self.vel_si_vec = MLP([input_dim, input_dim, 6])
        if regress_norm:
            self.vel_si_norm = MLP([input_dim, input_dim, 1])
    
    def forward(self, x_clu, mask_clu, batch_clu, dist_embedding, dist_embedding_norm):
        x_scene = self.pool(x_clu[mask_clu], batch_clu[mask_clu])
        vel_si_vec = self.vel_si_vec(x_scene)
        vel_si_norm = self.vel_si_norm(x_scene) if self.regress_norm else None
        return vel_si_vec * torch.mean(dist_embedding, dim=0), vel_si_norm * torch.mean(dist_embedding_norm, dim=0)


class GraphVS(nn.Module):
    def __init__(self, input_dim, pos_dim, hidden_dim=128, regress_norm=True):
        super(GraphVS, self).__init__()
        self.regress_norm = regress_norm
        self.encoder = Encoder(input_dim, hidden_dim, pos_dim) # [2, 128, 2]
        self.backbone = Backbone(hidden_dim, pos_dim)
        self.decoder = Decoder(hidden_dim, regress_norm)
    
    def init_hidden(self, size0):
        return self.backbone.temporal_aggr.init_hidden(size0)
    
    def forward(self, data: Union[GraphData, Batch], hidden: Optional[torch.Tensor] = None):
        x_cur = getattr(data, "x_cur")
        x_tar = getattr(data, "x_tar")
        pos_cur = getattr(data, "pos_cur")
        pos_tar = getattr(data, "pos_tar")

        depth_cur = getattr(data, "depth_cur")
        depth_tar = getattr(data, "depth_tar")

        l1_dense_edge_index_cur = getattr(data, "l1_dense_edge_index_cur")
        l1_dense_edge_index_tar = getattr(data, "l1_dense_edge_index_tar")

        l0_to_l1_edge_index_j_cur = getattr(data, "l0_to_l1_edge_index_j_cur")
        l0_to_l1_edge_index_i_cur = getattr(data, "l0_to_l1_edge_index_i_cur")
        l0_to_l1_edge_index_cur = torch.stack([l0_to_l1_edge_index_j_cur,
                                               l0_to_l1_edge_index_i_cur], dim=0)

        cluster_mask = getattr(data, "cluster_mask")
        cluster_centers_index = getattr(data, "cluster_centers_index")

        if not hasattr(data, "batch"):
            batch = None
        else:
            batch = getattr(data, "batch")

        if batch is None:
            batch = torch.zeros(x_cur.size(0)).long().to(x_cur.device)
        
        x_clu, dist_embedding, dist_embedding_norm = self.encoder(
            x_cur, x_tar, pos_cur, pos_tar, cluster_mask, 
            l0_to_l1_edge_index_cur, cluster_centers_index, depth_cur, depth_tar)
        pos_clu = pos_tar[cluster_centers_index]
        batch_clu = batch[cluster_centers_index]

        if hidden is None:
            hidden = self.init_hidden(getattr(data, "num_clusters").sum()).to(x_cur)
        hidden, x_clu = self.backbone(
            hidden, x_clu, pos_clu, l1_dense_edge_index_cur, l1_dense_edge_index_tar, batch_clu)
        
        vel_si_vec, vel_si_norm = self.decoder(x_clu, cluster_mask, batch_clu, dist_embedding, dist_embedding_norm)
        
        return vel_si_vec, vel_si_norm, hidden
    
    @classmethod
    def preprocess(cls, data: GraphData):
        return data
    
    @classmethod
    def postprocess(cls, raw_pred, data: GraphData):
        """From raw predictions to actual conduct velocity"""
        return postprocess(raw_pred, data, post_scale=True)
    
    @classmethod
    def objectives(cls, raw_pred, data: GraphData):
        weight = 1
        return objectives(raw_pred, data, gt_si=True, weight=weight)

    def get_parameter_groups(self):
        pg_wi_decay, pg_wo_decay = [], []
        for m in self.modules():
            if hasattr(m, "bias") and isinstance(m.bias, nn.Parameter):
                pg_wo_decay.append(m.bias)
            if isinstance(m, norm.GraphNorm):
                pg_wo_decay.append(m.weight)
                pg_wo_decay.append(m.mean_scale)
            elif hasattr(m, "weight") and isinstance(m.weight, nn.Parameter):
                pg_wi_decay.append(m.weight)
        return pg_wi_decay, pg_wo_decay

