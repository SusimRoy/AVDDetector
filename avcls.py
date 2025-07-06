from lightning.pytorch import LightningModule
from torch.optim import Adam
from transformers import VideoMAEImageProcessor, AutoModel, AutoConfig
import numpy as np
import torch
from audiossl.methods.atstframe.embedding import load_model,get_scene_embedding,get_timestamp_embedding
import audiossl.methods.atstframe.embedding as embedding
import torch.nn as nn
from torchmetrics.classification import AUROC
from models.mvit import MvitVideoEncoder
import torch.nn.functional as F

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
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.gate_fc = nn.Linear(in_feats, out_feats)
        
    def forward(self, feat1, feat2):
        # feat1, feat2: [B, 256]
        concat_feats = torch.cat([feat1, feat2], dim=-1)  
        gates = torch.sigmoid(self.gate_fc(concat_feats)) 
        fused = gates * feat1 + (1 - gates) * feat2      
        return fused

class AVClassifier(LightningModule):
    def __init__(self, lr, distributed=False):
        super(AVClassifier, self).__init__()
        self.lr = lr
        self.config = AutoConfig.from_pretrained("OpenGVLab/VideoMAEv2-Base", trust_remote_code=True)
        self.video_processor = VideoMAEImageProcessor.from_pretrained("OpenGVLab/VideoMAEv2-Base")
        self.videoextractor = AutoModel.from_pretrained('OpenGVLab/VideoMAEv2-Base', config=self.config, trust_remote_code=True).eval().requires_grad_(False)
        self.videofeatures = MvitVideoEncoder(v_cla_feature_in=256, temporal_size=112, mvit_type="mvit_v2_t")
        self.audioextractor = load_model("/home/csgrad/susimmuk/acmdeepfake/audio-extraction/atstframe_base.ckpt").eval().requires_grad_(False)
        self.cross_attention = CrossAttention(dim_q=256, dim_k=256, dim_out=256)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.conv1d_reduce = nn.Conv1d(768*embedding.N_BLOCKS, 256, kernel_size=1, bias=False)
        self.fc1 = nn.Linear(256+768, 1, bias=False)

        self.distributed = distributed
        self.init_weights()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.train_auc = AUROC(task="binary", sync_on_compute=True)
        self.test_probs = []
 
    def setup(self, stage: str):

        self.videoextractor.to(self.device)
        self.videofeatures.to(self.device)
        self.audioextractor.to(self.device)
        self.cross_attention.to(self.device)

    def init_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight)
        
        nn.init.kaiming_normal_(self.conv1d_reduce.weight)
    
        nn.init.kaiming_normal_(self.cross_attention.W_q.weight)
        nn.init.kaiming_normal_(self.cross_attention.W_k.weight)
        nn.init.kaiming_normal_(self.cross_attention.W_v.weight)
        
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
        afeats = F.interpolate(afeats, size=magfeatures.shape[2], mode="linear", align_corners=False)
        afeats = afeats.transpose(1, 2) # [B, 256, T2] -> [B, T2, 256]
        attn_feats = self.cross_attention(afeats, magfeatures.transpose(1,2)) # [B, T, 256] -> [B, T, 256]
        afeats += attn_feats
        afeats = afeats.mean(dim=1)
        fusionfeats = torch.cat([rgbfeatures, afeats], dim=1)
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
        optimizer = Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,  # or total steps if you prefer
            eta_min=1e-7  # final learning rate
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # step if you want per-batch
                "frequency": 1
            }
        }

    def on_test_start(self):
        self.test_probs = []

    def test_step(self, batch, batch_idx):
        x1, x2, x3, test_file = batch
        y_hat = self(x1, x2, x3)
        probs = torch.sigmoid(y_hat).detach().cpu().numpy().flatten()
        # test_file is usually a list of filenames
        if isinstance(test_file, str):
            test_file = [test_file]
        # results = []
        for fname, prob in zip(test_file, probs):
            self.test_probs.append((fname, float(prob)))
        # return results

    def on_test_epoch_end(self):
        if torch.distributed.is_initialized():
            gathered = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(gathered, self.test_probs)
            all_results = []
            for sublist in gathered:
                all_results.extend(sublist)
        else:
            all_results = self.test_probs

        unique_results = {}
        for fname, prob in all_results:
            base = fname.split('/')[-1]
            if not base.endswith('.mp4'):
                base = f"{base}.mp4"
            if base not in unique_results:
                unique_results[base] = prob

        if getattr(self, "trainer", None) is not None and self.trainer.is_global_zero:
            with open("prediction.txt", "w") as f:
                for base, prob in unique_results.items():
                    f.write(f"{base};{prob}\n")