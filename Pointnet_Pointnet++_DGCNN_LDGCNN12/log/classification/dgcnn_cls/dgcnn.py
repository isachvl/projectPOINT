import torch
import torch.nn as nn
import torch.nn.functional as F

# KNN и графовые функции
def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2,1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2,1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1,1,1)*num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2,1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1,1,k,1)
    feature = torch.cat((feature-x, x), dim=3).permute(0,3,1,2).contiguous()
    return feature

# Класс модели для совместимости с тренером
class DGCNN_Model(nn.Module):
    def __init__(self, num_class=40, k=20):
        super(DGCNN_Model, self).__init__()
        self.k = k

        # EdgeConv layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, 1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(128, 64, 1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, 1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(256, 256, 1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, 1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(0.2))

        # Fully connected
        self.fc1 = nn.Linear(2048, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256, bias=False)
        self.bn7 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.relu(self.bn6(self.fc1(x)))
        x = self.drop1(x)
        x = F.relu(self.bn7(self.fc2(x)))
        x = self.drop2(x)
        x = self.fc3(x)
        return x, None  # None вместо trans_feat для совместимости

# Функция для тренера
def get_model(num_class=40, normal_channel=False):
    return DGCNN_Model(num_class=num_class)

# Функция потерь для тренера
class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat=None):
        return F.cross_entropy(pred, target)
