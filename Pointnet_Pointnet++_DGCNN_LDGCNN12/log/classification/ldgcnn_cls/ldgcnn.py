import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================================
# KNN: ищет БЛИЖАЙШИХ соседей (k smallest distances)
# ======================================
def knn(x, k):
    """
    Input:
        x: point cloud (B, C, N)
        k: number of neighbors
    Output:
        idx: indices of k nearest neighbors (B, N, k)
    """
    # x: (B, C, N) → (B, N, C)
    x = x.transpose(2, 1).contiguous()
    # Compute pairwise squared distances: (B, N, N)
    # Using efficient matrix multiplication
    xx = torch.sum(x ** 2, dim=2, keepdim=True)  # (B, N, 1)
    yy = torch.sum(x ** 2, dim=2, keepdim=True).transpose(2, 1)  # (B, 1, N)
    dist = xx + yy - 2 * torch.bmm(x, x.transpose(2, 1))  # (B, N, N)
    # Prevent numerical issues
    dist = torch.clamp(dist, min=0.0)
    # Get indices of k smallest distances (nearest neighbors)
    _, idx = dist.topk(k=k, dim=-1, largest=False)  # (B, N, k)
    return idx

# ======================================
# Edge Feature: [x_j - x_i, x_i]
# ======================================
def get_edge_feature(x, k=20, idx=None):
    """
    Input:
        x: point features (B, C, N)
        k: number of neighbors
        idx: precomputed neighbor indices (optional)
    Output:
        edge_features: (B, 2*C, N, k)
    """
    B, C, N = x.size()
    
    if idx is None:
        idx = knn(x, k=k)  # (B, N, k)
    
    # Add base index for batch flattening
    idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1) * N
    idx = idx + idx_base  # (B, N, k)
    idx = idx.view(-1)  # (B*N*k)

    # Flatten x for indexing: (B, C, N) → (B*N, C)
    x_flat = x.transpose(2, 1).contiguous().view(B * N, C)

    # Gather neighbors: (B*N*k, C)
    neighbors = x_flat[idx, :]  # (B*N*k, C)
    neighbors = neighbors.view(B, N, k, C)  # (B, N, k, C)

    # Central points: (B, N, 1, C)
    center = x.transpose(2, 1).unsqueeze(2)  # (B, N, 1, C)

    # Edge features: [x_j - x_i, x_i]
    edge_feat = torch.cat([neighbors - center, center], dim=3)  # (B, N, k, 2C)
    edge_feat = edge_feat.permute(0, 3, 1, 2).contiguous()  # (B, 2C, N, k)
    
    return edge_feat

# ======================================
# LDGCNN Model (Local Dense Graph CNN)
# ======================================
class LDGCNN(nn.Module):
    def __init__(self, num_class=40, k=20, normal_channel=False):
        super(LDGCNN, self).__init__()
        self.k = k
        self.normal_channel = normal_channel

        # EdgeConv layers (with dense connections)
        self.conv1 = nn.Sequential(
            nn.Conv2d(2 * 3, 64, kernel_size=1, bias=False),  # 3D points → 6
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(2 * (64 + 3), 64, kernel_size=1, bias=False),  # [x, f1] → 67 → 134
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(2 * (64 + 64 + 3), 64, kernel_size=1, bias=False),  # [x, f1, f2] → 131 → 262
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(2 * (64 + 64 + 64 + 3), 128, kernel_size=1, bias=False),  # [x, f1, f2, f3] → 195 → 390
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2)
        )

        # Final aggregation: x (3) + f1 (64) + f2 (64) + f3 (64) + f4 (128) = 323
        self.conv5 = nn.Sequential(
            nn.Conv1d(3 + 64 + 64 + 64 + 128, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2)
        )

        # Classifier
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, x):
        # x: (B, C, N), C = 3 or 6
        if self.normal_channel:
            # Use only xyz (drop normals) — LDGCNN typically uses only coords
            x = x[:, :3, :]  # (B, 3, N)
        else:
            # Ensure only xyz
            x = x[:, :3, :]  # (B, 3, N)

        B, C, N = x.shape  # C == 3

        # === EdgeConv 1 ===
        edge1 = get_edge_feature(x, k=self.k)  # (B, 6, N, k)
        f1 = self.conv1(edge1).max(dim=-1, keepdim=False)[0]  # (B, 64, N)

        # === EdgeConv 2: [x, f1] ===
        x2 = torch.cat([x, f1], dim=1)  # (B, 3+64, N)
        edge2 = get_edge_feature(x2, k=self.k)  # (B, 2*67, N, k) = (B, 134, N, k)
        f2 = self.conv2(edge2).max(dim=-1, keepdim=False)[0]  # (B, 64, N)

        # === EdgeConv 3: [x, f1, f2] ===
        x3 = torch.cat([x, f1, f2], dim=1)  # (B, 3+64+64, N) = (B, 131, N)
        edge3 = get_edge_feature(x3, k=self.k)  # (B, 262, N, k)
        f3 = self.conv3(edge3).max(dim=-1, keepdim=False)[0]  # (B, 64, N)

        # === EdgeConv 4: [x, f1, f2, f3] ===
        x4 = torch.cat([x, f1, f2, f3], dim=1)  # (B, 3+64*3, N) = (B, 195, N)
        edge4 = get_edge_feature(x4, k=self.k)  # (B, 390, N, k)
        f4 = self.conv4(edge4).max(dim=-1, keepdim=False)[0]  # (B, 128, N)

        # === Global feature: concatenate all + original x ===
        global_feat = torch.cat([x, f1, f2, f3, f4], dim=1)  # (B, 3+64+64+64+128, N) = (B, 323, N)
        global_feat = self.conv5(global_feat)  # (B, 1024, N)

        # Global max pooling
        global_feat = F.adaptive_max_pool1d(global_feat, 1).squeeze(-1)  # (B, 1024)

        # Classifier
        x_feat = F.leaky_relu(self.bn1(self.fc1(global_feat)), negative_slope=0.2)
        x_feat = self.drop1(x_feat)
        x_feat = F.leaky_relu(self.bn2(self.fc2(x_feat)), negative_slope=0.2)
        x_feat = self.drop2(x_feat)
        logits = self.fc3(x_feat)  # (B, num_class)

        return logits, global_feat  # compatible with typical trainers

# ======================================
# Interface for trainer scripts
# ======================================
def get_model(num_class=40, normal_channel=False, **kwargs):
    return LDGCNN(num_class=num_class, normal_channel=normal_channel)

def get_loss(**kwargs):
    return nn.CrossEntropyLoss()