import torch
import torch.nn as nn
import torch.nn.functional as F

def knn(x, k):
    # x: (B, C, N)
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # (B, N, N)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)     # (B, 1, N)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # = -||xi - xj||^2
    idx = pairwise_distance.topk(k=k, dim=-1, largest=True)[1]  # (B, N, k)
    return idx

def get_graph_feature(x, k=20, idx=None):
    batch_size, _, num_points = x.size()
    if idx is None:
        idx = knn(x, k=k)  # (B, N, k)

    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    x = x.transpose(2, 1).contiguous()  # (B, N, C)
    feature = x.view(batch_size * num_points, -1)[idx, :]  # (B*N*k, C)
    feature = feature.view(batch_size, num_points, k, -1)  # (B, N, k, C)
    x = x.view(batch_size, num_points, 1, -1).repeat(1, 1, k, 1)  # (B, N, k, C)

    feature = torch.cat([feature - x, x], dim=3)  # (B, N, k, 2C)
    feature = feature.permute(0, 3, 1, 2).contiguous()  # (B, 2C, N, k)
    return feature

class DGCNN_Model(nn.Module):
    def __init__(self, num_class=40, k=20):
        super(DGCNN_Model, self).__init__()
        self.k = k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2)
        )

        # Classifier 
        self.fc1 = nn.Linear(1024 * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256, bias=False)
        self.bn7 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, x):
        # x: (B, C, N), usually C=3
        batch_size = x.size(0)

        x = get_graph_feature(x, k=self.k)    # (B, 6, N, k) — only xyz
        x = self.conv1(x)                      # (B, 64, N, k)
        x1 = x.max(dim=-1, keepdim=False)[0]   # (B, 64, N)

        x = get_graph_feature(x1, k=self.k)   # (B, 128, N, k)
        x = self.conv2(x)                      # (B, 64, N, k)
        x2 = x.max(dim=-1, keepdim=False)[0]   # (B, 64, N)

        x = get_graph_feature(x2, k=self.k)   # (B, 128, N, k)
        x = self.conv3(x)                      # (B, 128, N, k)
        x3 = x.max(dim=-1, keepdim=False)[0]   # (B, 128, N)

        x = get_graph_feature(x3, k=self.k)   # (B, 256, N, k)
        x = self.conv4(x)                      # (B, 256, N, k)
        x4 = x.max(dim=-1, keepdim=False)[0]   # (B, 256, N)

        x = torch.cat([x1, x2, x3, x4], dim=1)  # (B, 512, N)
        x = self.conv5(x)                        # (B, 1024, N)

        # Global features: max + avg pooling
        x_max = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)    # (B, 1024)
        x_avg = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)    # (B, 1024)
        x = torch.cat([x_max, x_avg], dim=1)                        # (B, 2048)

        # Classifier — ВЕЗДЕ LeakyReLU!
        x = F.leaky_relu(self.bn6(self.fc1(x)), negative_slope=0.2)
        x = self.drop1(x)
        x = F.leaky_relu(self.bn7(self.fc2(x)), negative_slope=0.2)
        x = self.drop2(x)
        x = self.fc3(x)

        return x, None   

 
def get_model(num_class=40, normal_channel=False):
    return DGCNN_Model(num_class=num_class)

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat=None):
        return F.cross_entropy(pred, target)