import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data

def conv_layer(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_dim), nn.ReLU(True))


def linear_layer(in_dim, out_dim, bias=False):
    return nn.Sequential(nn.Linear(in_dim, out_dim, bias),
                         nn.BatchNorm1d(out_dim), nn.ReLU(True))

def pool_visual_features(visual_features, pooling_type='max'):
    """
    Pool the 3D visual features to 2D.
    visual_features: Tensor of shape [b, 576, 768]
    pooling_type: 'max' or 'avg'
    """
    if pooling_type == 'max':
        pooled, _ = torch.max(visual_features, dim=1)
    elif pooling_type == 'avg':
        pooled = torch.mean(visual_features, dim=1)
    else:
        raise ValueError("Unsupported pooling type. Choose 'max' or 'avg'.")
    return pooled

def convert_to_logits(tensor, epsilon=1e-6):
    # Convert to logits
    tensor = torch.clamp(tensor, epsilon, 1 - epsilon)
    logits = torch.log(tensor / (1 - tensor))

    return logits


class CoordConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1,
                 stride=1):
        super().__init__()
        self.conv1 = conv_layer(in_channels + 2, out_channels, kernel_size,
                                padding, stride)

    def add_coord(self, input):
        b, _, h, w = input.size()
        x_range = torch.linspace(-1, 1, w, device=input.device)
        y_range = torch.linspace(-1, 1, h, device=input.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([b, 1, -1, -1])
        x = x.expand([b, 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        input = torch.cat([input, coord_feat], 1)
        return input

    def forward(self, x):
        x = self.add_coord(x)
        x = self.conv1(x)
        return x


class DimensionalReduction(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DimensionalReduction, self).__init__()
        self.reduce = nn.Sequential(
            conv_layer(in_channel, out_channel, 3, padding=1),
            conv_layer(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):
        return self.reduce(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



class AttentionMutualFusion(nn.Module):
    def __init__(self, input_dim=768, embed_dim=768):
        super(AttentionMutualFusion, self).__init__()
        self.attention_weights = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, tensor1, tensor2):
        # Compute attention weights
        attn1 = self.attention_weights(tensor1.squeeze(1))
        attn2 = self.attention_weights(tensor2.squeeze(1))
        # Apply attention weights
        fused_tensor = attn1 * tensor1 + attn2 * tensor2
        return fused_tensor
    

class TextualRefinement(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(TextualRefinement, self).__init__()
        self.camo_in_overalldesc = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.obj_in_camodesc = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)


        self.add_norm1 = nn.LayerNorm(embed_dim)
        self.add_norm2 = nn.LayerNorm(embed_dim)

        self.mlp1 = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        self.mutual_fusion = AttentionMutualFusion(input_dim=768, embed_dim=768)

        self.obj_projector = nn.Linear(input_dim=768, embed_dim=768)
        self.camo_projector = nn.Linear(input_dim=768, embed_dim=17)
        # alternative none-learnable projector
        # self.projector = nn.Sequential(
        #     nn.Linear(768, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 17)
        # )
        
        
    def forward(self, textual_dist):
        camo_in_overalldesc_output, _ = self.camo_in_overalldesc(query=textual_dist['overall_s'], key=textual_dist['camo_w'], value=textual_dist['camo_w'])
        obj_in_camodesc_output, _ = self.obj_in_camodesc(query=textual_dist['camo_s'], key=textual_dist['overall_w'], value=textual_dist['overall_w'])
        
        # add norm
        camo_in_overalldesc_output = camo_in_overalldesc_output + textual_dist['overall_s']
        obj_in_camodesc_output = obj_in_camodesc_output + textual_dist['camo_s']

        camo_in_overalldesc_output = self.add_norm1(camo_in_overalldesc_output)
        obj_in_camodesc_output = self.add_norm2(obj_in_camodesc_output)

        # camo_in_overalldesc_output = self.layer_norm1(camo_in_overalldesc_output + overall_desc)
        # obj_in_camodesc_output = self.layer_norm2(obj_in_camodesc_output + camo_desc)
        
        obj_desc = obj_in_camodesc_output * textual_dist['overall_s']
        camo_desc = camo_in_overalldesc_output * textual_dist['camo_s']

        
        obj_desc = torch.cat([obj_desc, textual_dist['overall_s']], dim=1)
        camo_desc = torch.cat([camo_desc, textual_dist['camo_s']], dim=1)

        obj_desc = self.mlp1(obj_desc)
        camo_desc = self.mlp2(camo_desc)


        refined_desc = self.mutual_fusion(obj_desc, camo_desc)
        
        projected_obj = self.obj_projector(obj_desc)
        projected_camo = self.camo_projector(camo_desc)



        return refined_desc, projected_obj, projected_camo


class ForegroundBackgroundAlignment(nn.Module):
    def __init__(self, binary_mask_threshold=0.5):
        super(ForegroundBackgroundAlignment, self).__init__()
        self.binary_mask_threshold = binary_mask_threshold

    def forward(self, predict_mask, img):
        # Ensure predict_mask and img have a batch dimension
        if predict_mask.dim() == 3:
            predict_mask = predict_mask.unsqueeze(0)
        if img.dim() == 3:
            img = img.unsqueeze(0)
            
        # Upsample the mask to the same size as the image
        predict_mask = F.interpolate(predict_mask, size=img.shape[-2:], mode='bilinear', align_corners=False)
        
        # Threshold the mask
        predict_mask = (predict_mask > self.binary_mask_threshold).float()

        # Generate the foreground and background images from the mask and original image
        im_masked_fg = predict_mask * img
        im_masked_bg = (1 - predict_mask) * img

        return im_masked_fg, im_masked_bg


class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads, num_features=3):
        super(CrossAttentionFusion, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_features)])  # List of LayerNorms

    def forward(self, features):
        normalized_features = [norm(feature) for norm, feature in zip(self.norms, features)]
        kv = torch.cat(normalized_features, dim=1)  # Shape: [b, 1728, 768]
        query = normalized_features[2]  # Shape: [b, 576, 768]
        attn_output, _ = self.attention(query=query, key=kv, value=kv)
        output = attn_output + query  # Shape: [b, 576, 768]
        return output


class FixationEstimation(nn.Module):
    def __init__(self, embed_dim, num_heads, num_decoder_layers, dim_feedforward, output_map_size):
        super(FixationEstimation, self).__init__()
        self.pos_encoder = PositionalEncoding(embed_dim)
        self.fusion = CrossAttentionFusion(embed_dim, num_heads)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=dim_feedforward)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.intermediate_linear = nn.Linear(embed_dim, output_map_size * output_map_size)
        self.aggregate_conv = nn.Conv2d(in_channels=576, out_channels=1, kernel_size=1)
        self.tensor_out_conv = nn.Conv1d(in_channels=576, out_channels=768, kernel_size=1)
        self.reshape_size = output_map_size
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        
        # Learnable memory initialization
        self.learnable_memory = nn.Parameter(torch.randn(1, embed_dim), requires_grad=True)

    def forward(self, feature_list):
        fused_features = self.fusion(feature_list)
        fused_features = fused_features.permute(1, 0, 2)  # Shape: [sequence_length, batch_size, feature_size]
        fused_features = self.pos_encoder(fused_features)  # Add positional encoding

        # self-attention mode
        out_fix = self.transformer_decoder(fused_features, fused_features)

        # Reshape and project the output to the desired fixation map size
        out_fix = self.intermediate_linear(out_fix)
        out_fix = out_fix.view(-1, 576, self.reshape_size, self.reshape_size)  # Shape: [b, 576, 24, 24]
        out_tensor = out_fix.view(-1, 576, self.reshape_size * self.reshape_size)
        out_tensor = self.tensor_out_conv(out_tensor).transpose(1, 2)  # Shape: [b, 576, 768]
        # # Adding skip connection
        out_fix = self.aggregate_conv(out_fix)  # Shape: [b, 1, 24, 24]
        out_fix = self.upsample(out_fix)  # Shape: [b, 1, 96, 96]

        return out_fix, out_tensor


class WeightedAttributePrediction(nn.Module):
    def __init__(self, num_tokens, feature_dim, num_attr, deeper_layer_weight=0.5):
        super(AttributePrediction, self).__init__()
        self.deeper_layer_weight = deeper_layer_weight
        self.middle_layer_weight = (1 - deeper_layer_weight) / 2
        self.shallower_layer_weight = (1 - deeper_layer_weight) / 2

        # Initial dimension reduction
        self.init_reduce_dim = nn.Linear(feature_dim, 256)

        # Further dimension reduction
        self.reduce_dim = nn.Linear(num_tokens * 256 * 3, 512)

        # Fully connected layers
        self.fc1 = nn.Linear(512, 256)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_attr)

    def forward(self, tensors):
        # Apply weights and initial reduction to each tensor
        reduced_tensors = [self.init_reduce_dim(t * weight) for t, weight in zip(tensors, 
                        [self.shallower_layer_weight, self.middle_layer_weight, self.deeper_layer_weight])]

        # Concatenate the tensors along the feature dimension
        concatenated = torch.cat(reduced_tensors, dim=-1)  # Size: [b, 576, 256*3]
        
        # Further reduce the dimensionality of the concatenated tensor
        reduced = self.reduce_dim(concatenated.view(concatenated.size(0), -1))

        # Apply linear layers and other components
        attr_ctrb = self.fc1(reduced)
        attr_ctrb = self.batch_norm1(attr_ctrb)
        attr_ctrb = self.relu(attr_ctrb)
        attr_ctrb = self.dropout(attr_ctrb)
        attr_ctrb = self.fc2(attr_ctrb)

        return attr_ctrb


class AttributePrediction(nn.Module):
    def __init__(self, num_tokens, feature_dim, num_attr):
        super(AttributePrediction, self).__init__()

        # Initial dimension reduction
        self.init_reduce_dim = nn.Linear(feature_dim, 256)

        # Further dimension reduction
        self.reduce_dim = nn.Linear(num_tokens * 256 * 3, 512)

        # Fully connected layers
        self.fc1 = nn.Linear(512, 256)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_attr)

    def forward(self, tensors):
        
        # Further reduce the dimensionality of the concatenated tensor
        reduced = self.reduce_dim(tensors[2])

        # Apply linear layers and other components
        attr_ctrb = self.fc1(reduced)
        attr_ctrb = self.batch_norm1(attr_ctrb)
        attr_ctrb = self.relu(attr_ctrb)
        attr_ctrb = self.dropout(attr_ctrb)
        attr_ctrb = self.fc2(attr_ctrb)

        return attr_ctrb


class CrossModalTransformerLayer(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super(CrossModalTransformerLayer, self).__init__()
        
        self.self_attn_vision = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads)
        self.cross_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads)

    def forward(self, semantic_features, vision_features):
    
        semantic_features_expanded = semantic_features.unsqueeze(1)
        vision_features = vision_features.permute(1, 0, 2)
        vision_features_attended, _ = self.self_attn_vision(vision_features, vision_features, vision_features)
        semantic_features_transformed, attn_update = self.cross_attn(vision_features_attended, semantic_features_expanded, semantic_features_expanded) # [b, sequence_length, feature_dim]
        
        return semantic_features_transformed, attn_update

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Expecting x of shape [b, sequence_length, channel] for vision or [b, channel] for semantic features
        original_shape = x.shape
        
        y = self.avg_pool(x).squeeze(-1)
        y = self.fc(y).view(original_shape[0], original_shape[-1], -1)
        return x * y.expand_as(x)

class SemanticHierarchicalEmbedding(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super(SemanticHierarchicalEmbedding, self).__init__()
        self.cross_attn_0  = CrossModalTransformerLayer(feature_dim, num_heads)
        self.cross_attn_1 = CrossModalTransformerLayer(feature_dim, num_heads)
        self.cross_attn_2 = CrossModalTransformerLayer(feature_dim, num_heads)

        self.se_semantic_shallow = SEBlock(feature_dim)
        self.se_semantic_deep = SEBlock(feature_dim)

        self.layer_norm1 = nn.LayerNorm(feature_dim)
        self.layer_norm2 = nn.LayerNorm(feature_dim)


    def forward(self, semantic_features, vision_features, attn_update_w=[0.2, 0.3, 0.5]):
        x_0, attn_update0 = self.cross_attn_0(semantic_features, vision_features[0])
        
        x_1, attn_update1 = self.cross_attn_1(semantic_features, vision_features[1])
        x_1 = x_1 + x_0
        x_1 = self.layer_norm1(x_1)
        x_1 = self.se_semantic_shallow(x_1)

        x_2, attn_update2 = self.cross_attn_2(semantic_features, vision_features[1])
        x_2 = x_2 + x_1
        x_2 = self.layer_norm2(x_2)
        x_2 = self.se_semantic_shallow(x_2)

        attn = attn_update0 * attn_update_w[0] + attn_update1 * attn_update_w[1] + attn_update2 * attn_update_w[2]
        
        x_2 = x_2 + attn * vision_features[2]
        
        return x_2

class ProjectionNetwork(nn.Module):
    def __init__(self, input_dim, proj_dim, hidden_dim=None):
        super(ProjectionNetwork, self).__init__()
        if hidden_dim is None:
            hidden_dim = (input_dim + proj_dim) // 2  # A heuristic for hidden dimension size

        # Define the layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, proj_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class ConvBR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1):
        super(ConvBR, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class Projector(nn.Module):
    def __init__(self, word_dim=768, in_dim=768, kernel_size=3, num_attr=17):
        super().__init__()
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        # visual projector
        self.vis = nn.Sequential(  # spatical resolution times 4
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(in_dim , in_dim * 2, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(in_dim * 2, in_dim, 3, padding=1),
            nn.Conv2d(in_dim, in_dim, 1))
        
        # textual projector
        self.attr_proj = nn.Linear(num_attr, word_dim)
        out_dim = 1 * word_dim * kernel_size * kernel_size + 1
        self.txt = nn.Linear(word_dim, out_dim)

        # output projector
        self.out_proj = nn.Sequential(
            ConvBR(in_dim, 128, 3, padding=1),
            ConvBR(128, 64, 3, padding=1),
            nn.Conv2d(64, 1, 1))

    def forward(self, x, attr, use_attr=False):
        '''
            x: b, vis_dim, 24, 24
        '''
        if use_attr:
            x = self.vis(x) # b, 768, 96, 96
            B, C, H, W = x.size()
            x = x.reshape(1, B * C, H, W) # 1, b*768, 96, 96
            # txt: b, (768*3*3 + 1) -> b, 768, 3, 3 / b 
            attr = self.attr_proj(attr)
            attr = self.txt(attr)
            weight, bias = attr[:, :-1], attr[:, -1]
            weight = weight.reshape(B, C, self.kernel_size, self.kernel_size)
            # Conv2d - 1, b*768, 96, 96 -> 1, b, 96, 96
            out = F.conv2d(x,
                        weight,
                        padding=self.kernel_size // 2,
                        groups=weight.size(0),
                        bias=bias)
            out = out.transpose(0, 1)   # b, 1, 96, 96
            return out
        else:
            x = self.vis(x) # b, 768, 96, 96
            B, C, H, W = x.size()
            # b, 768, 96, 96 -> b, 1, 96, 96
            out = self.out_proj(x)
            # b, 1, 96, 96
            return out
            


class TransformerDecoder(nn.Module):
    def __init__(self,
                 num_layers,
                 d_model,
                 nhead,
                 dim_ffn,
                 dropout,
                 return_intermediate=False):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model=d_model,
                                    nhead=nhead,
                                    dim_feedforward=dim_ffn,
                                    dropout=dropout) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.return_intermediate = return_intermediate

    @staticmethod
    def pos1d(d_model, length):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                              -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe.unsqueeze(1)  # n, 1, 512

    @staticmethod
    def pos2d(d_model, height, width):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(
            torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)

        return pe.reshape(-1, 1, height * width).permute(2, 1, 0)  # hw, 1, 512

    def forward(self, vis):
        '''
            vis: b, 512, h, w
            txt: b, L, 512
            
        '''
        B, C, H, W = vis.size()
        # position encoding
        vis_pos = self.pos2d(C, H, W)
        # reshape & permute
        vis = vis.reshape(B, C, -1).permute(2, 0, 1)
        # forward
        output = vis
        intermediate = []
        for layer in self.layers:
            output = layer(output, vis_pos)
            if self.return_intermediate:
                # HW, b, 512 -> b, 512, HW
                intermediate.append(self.norm(output).permute(1, 2, 0))

        if self.norm is not None:
            # HW, b, 512 -> b, 512, HW
            output = self.norm(output).permute(1, 2, 0)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
                # [output1, output2, ..., output_n]
                return intermediate
            else:
                # b, 512, HW
                return output
        return output

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # Normalization Layer
        self.self_attn_norm = nn.LayerNorm(d_model)
        # Attention Layer
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # FFN
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_feedforward),
                                 nn.ReLU(True), nn.Dropout(dropout),
                                 nn.Linear(dim_feedforward, d_model))
        # LayerNorm & Dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos.to(tensor.device)

    def forward(self, vis, vis_pos):
        '''
            vis: 24*24, b, 512
            vis_pos: 24*24, 1, 512
        '''
        # Self-Attention
        vis2 = self.norm1(vis)
        q = k = self.with_pos_embed(vis2, vis_pos)
        vis2 = self.self_attn(q, k, value=vis2)[0]
        vis2 = self.self_attn_norm(vis2)
        vis = vis + self.dropout1(vis2)
        # FFN
        vis2 = self.norm2(vis)
        vis2 = self.ffn(vis2)
        vis = vis + self.dropout2(vis2)
        return vis


def d3_to_d4(self, t):
        n, hw, c = t.size()
        if hw % 2 != 0:
            t = t[:, 1:]
        h = w = int(math.sqrt(hw))
        return t.transpose(1, 2).reshape(n, c, h, w)

