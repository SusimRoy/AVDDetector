import timm
from lightning.pytorch import LightningModule
from torch.nn import CrossEntropyLoss, L1Loss
from torch.optim import Adam
from transformers import VideoMAEImageProcessor, AutoModel, AutoConfig
import numpy as np
import torch
from audiossl.methods.atstframe.embedding import load_model,get_scene_embedding,get_timestamp_embedding
import audiossl.methods.atstframe.embedding as embedding
import torch.nn as nn
from torchmetrics.classification import AUROC
from models.c3d import C3DVideoEncoder


class FusionModule(nn.Module):
    def __init__(self, dim):
        super(FusionModule, self).__init__()
        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        nn.init.kaiming_normal_(self.W_q.weight)
        nn.init.kaiming_normal_(self.W_k.weight)
        nn.init.kaiming_normal_(self.W_v.weight)

    def forward(self, x):
        dim = x.shape[1]
        batch_size = x.shape[0]
        x_seq = x.view(batch_size, dim, 1)  # [batch, seq_len, feature_dim=1]
        q = self.W_q(x).view(batch_size, dim, 1)  # [1, dim, 1]
        k = self.W_k(x).view(batch_size, dim, 1)  # [1, dim, 1]
        v = self.W_v(x).view(batch_size, dim, 1)  # [1, dim, 1]

        # Compute affinity matrix: [768, 768]
        # q: [1, 768, 1], k: [1, 768, 1] -> [1, 768, 768]
        attn_scores = torch.bmm(q, k.transpose(1, 2)) / np.sqrt(1)
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [1, 768, 768]

        # Weighted sum of v: [1, 768, 1]
        attended = torch.bmm(attn_weights, v)  # [1, 768, 1]
        attended = attended.view(batch_size, dim)
        return attended

class CrossAttention(nn.Module):
    def __init__(self, dim_q, dim_k, dim_out):
        super(CrossAttention, self).__init__()
        self.dim_out = dim_out
        self.W_q = nn.Linear(dim_q, dim_out, bias=False)
        self.W_k = nn.Linear(dim_k, dim_out, bias=False)
        self.W_v = nn.Linear(dim_k, dim_out, bias=False)
        self.scale = dim_out ** -0.5
        
        # nn.init.kaiming_normal_(self.W_q.weight)
        # nn.init.kaiming_normal_(self.W_k.weight)
        # nn.init.kaiming_normal_(self.W_v.weight)
    
    def forward(self, query, key_value):
        # query: [B, T, dim_q] (audio features)
        # key_value: [B, dim_k, T] (magnitude features)
        
        # Transpose key_value to match query dimensions        
        batch_size, seq_len_q, _ = query.shape
        _, seq_len_kv, _ = key_value.shape
        
        # Project to query, key, value
        Q = self.W_q(query)  # [B, T, dim_out]
        K = self.W_k(key_value)  # [B, T, dim_out]
        V = self.W_v(key_value)  # [B, T, dim_out]
        
        # Compute attention scores
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale  # [B, T, T]
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Apply attention to values
        attended = torch.bmm(attn_weights, V)  # [B, T, dim_out]
        
        return attended

class GatedFusion(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        # Learnable linear layer for gating (can include both features)
        self.gate_fc = nn.Linear(feat_dim * 2, feat_dim)
        
    def forward(self, feat1, feat2):
        # feat1, feat2: [B, 256]
        concat_feats = torch.cat([feat1, feat2], dim=-1)  # [B, 512]
        gates = torch.sigmoid(self.gate_fc(concat_feats)) # [B, 256], values in [0,1]
        # Gate feat1, use (1-gate) for feat2 (complementary)
        fused = gates * feat1 + (1 - gates) * feat2       # [B, 256]
        return fused

class AVClassifier(LightningModule):
    def __init__(self, lr, n_channels, distributed=False):
        super(AVClassifier, self).__init__()
        self.lr = lr
        self.config = AutoConfig.from_pretrained("OpenGVLab/VideoMAEv2-Large", trust_remote_code=True)
        self.video_processor = VideoMAEImageProcessor.from_pretrained("OpenGVLab/VideoMAEv2-Large")
        self.videoextractor = AutoModel.from_pretrained('OpenGVLab/VideoMAEv2-Large', config=self.config, trust_remote_code=True).eval().requires_grad_(False)
        # videoextractor.eval()
        # for param in videoextrasctor.parameters():
        #     param.requires_grad = False
        # self.add_module('videoextractor', videoextractor)
        self.videofeatures = C3DVideoEncoder(n_features=(64, 96, 128, 128), v_cla_feature_in=256)  # Initialize C3D encoder
        # self.gated_fusion = GatedFusion(feat_dim=256)  # Initialize gated fusion module
        # Initialize audio extractor as buffer
        self.audioextractor = load_model("/home/csgrad/susimmuk/acmdeepfake/audio-extraction/atstframe_base.ckpt").eval().requires_grad_(False)
        self.videofeatures = MvitVideoEncoder(v_cla_feature_in=v_cla_feature_in, temporal_size=temporal_size, mvit_type="mvit_v2_s")
        # audioextractor.eval()
        # for param in audioextractor.parameters():
        #     param.requires_grad = False
        # self.add_module('audioextractor', audioextractor)
        self.cross_attention = CrossAttention(dim_q=256, dim_k=256, dim_out=256)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.conv1d_reduce = nn.Conv1d(768*embedding.N_BLOCKS, 256, kernel_size=1, bias=False)
        self.fc1 = nn.Linear(256+768, 1, bias=False)
        # self.conv2 = nn.Conv1d(786, 256, kernel_size=1, bias=False)
        # self.fc2 = nn.Linear(768, 256, bias=False)

        self.distributed = distributed
        self.init_weights()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.train_auc = AUROC(task="binary", sync_on_compute=True)

    def setup(self, stage: str):

        self.videoextractor.to(self.device)
        self.audioextractor.to(self.device)

    def init_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight)
        
        # Initialize conv1d_reduce layer
        nn.init.kaiming_normal_(self.conv1d_reduce.weight)
        # nn.init.kaiming_normal_(self.fc2.weight)
        
        # Initialize cross attention weights
        nn.init.kaiming_normal_(self.cross_attention.W_q.weight)
        nn.init.kaiming_normal_(self.cross_attention.W_k.weight)
        nn.init.kaiming_normal_(self.cross_attention.W_v.weight)
        
        # Initialize C3D video encoder weights
        for module in self.videofeatures.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def process_video_in_chunks(self, video, chunk_size=16):
        num_frames = video.shape[1]
        num_chunks = num_frames // chunk_size
        all_features = []
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size    
            chunk = video[:, start_idx:end_idx]
            chunk = chunk.permute(0, 2, 1, 3, 4)
            with torch.no_grad():
                outputs = self.videoextractor(chunk.float())
            all_features.append(outputs)
            
        all_features = torch.stack(all_features)
        all_features = all_features.type_as(video)
        final_features = all_features.mean(dim=0)
        return final_features

    def forward(self, vinp, maginp, audioinp):
        # input : vinp: [B, T1, C, H, W], maginp: [B, T1, C, H, W], audioinp: [B, T2, c2]
        diff = maginp - vinp
        video = torch.cat([vinp, diff], dim=2)
        vinp_resized = torch.nn.functional.interpolate(vinp.view(-1, vinp.shape[-3], vinp.shape[-2], vinp.shape[-1]), size=(224, 224), mode='bilinear', align_corners=False).reshape(vinp.shape[0], vinp.shape[1], vinp.shape[2], 224, 224)
        rgbfeatures = self.process_video_in_chunks(vinp_resized)
        video = video.permute(0, 2, 1, 3, 4) # [B, T1, C, H, W] -> [B, C, T1, H, W]
        magfeatures = self.videofeatures(video) # [B, C, T1, H, W] -> [B, 256, T1]
        with torch.no_grad():
            afeats,_ = get_timestamp_embedding(audioinp, self.audioextractor)
        afeats = afeats.transpose(1, 2)  # [B, T2, 768*12] -> [B, 768*12, T2]
        afeats = self.conv1d_reduce(afeats)  # [B, 768*12, T2] -> [B, 256, T2]
        afeats = afeats.transpose(1, 2)   # [B, 256, T2] -> [B, T2, 256]
        target_T = magfeatures.shape[2]  # T1 from magfeatures
        current_T = afeats.shape[1]      # T2 from afeats
        if current_T < target_T:
            last_timestep = afeats[:, -1:, :]
            repeat_count = target_T - current_T
            repeated_timesteps = last_timestep.repeat(1, repeat_count, 1)
            afeats = torch.cat([afeats, repeated_timesteps], dim=1)
        attn_feats = self.cross_attention(afeats, magfeatures.transpose(1,2))  # [B, T, 256] -> [B, T, 256]
        # rgbfeatures = self.fc2(rgbfeatures)
        fusionfeats = torch.cat([rgbfeatures, attn_feats.mean(dim=1)], dim=1)
        y_hat = self.fc1(fusionfeats)  
        return y_hat

    def training_step(self, batch, batch_idx):
        x1, x2, x3, y = batch
        y_hat = self(x1, x2, x3)  # [batch, 1]
        loss = self.loss_fn(y_hat.squeeze(-1), y.float())
        probs = torch.sigmoid(y_hat).detach()
        self.train_auc.update(probs, y)
        self.log("train_loss", loss, prog_bar=True)
        self.train_loss_sum += loss.item()
        self.train_loss_count += 1

        return loss


    def on_train_start(self):
        self.train_loss_sum = 0.0
        self.train_loss_count = 0

    def on_train_epoch_end(self):
        auc = self.train_auc.compute()   # ← gathers across all ranks
        self.log("train_epoch_auc", auc, prog_bar=True, sync_dist=True)
        avg_loss = self.train_loss_sum / self.train_loss_count
        self.log("train_epoch_loss", avg_loss, sync_dist=True, prog_bar=True)
        self.train_loss_sum = 0.0
        self.train_loss_count = 0
        self.train_auc.reset()

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay =  0.0001)
        return [optimizer]