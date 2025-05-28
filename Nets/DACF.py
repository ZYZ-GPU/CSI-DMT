import torch
import torch.nn as nn
import torch.nn.functional as F


class DACF(nn.Module):  # Dual Attention Cross Fusion
    def __init__(self):
        super(DACF, self).__init__()

        self.conv1_spatial = nn.Conv2d(2, 1, 3, stride=1, padding=1, groups=1)
        self.conv2_spatial = nn.Conv2d(1, 1, 3, stride=1, padding=1, groups=1)

        self.avg1 = nn.Conv2d(32, 8, 1, stride=1, padding=0)
        self.avg2 = nn.Conv2d(32, 8, 1, stride=1, padding=0)
        self.max1 = nn.Conv2d(32, 8, 1, stride=1, padding=0)
        self.max2 = nn.Conv2d(32, 8, 1, stride=1, padding=0)

        self.avg11 = nn.Conv2d(8, 32, 1, stride=1, padding=0)
        self.avg22 = nn.Conv2d(8, 32, 1, stride=1, padding=0)
        self.max11 = nn.Conv2d(8, 32, 1, stride=1, padding=0)
        self.max22 = nn.Conv2d(8, 32, 1, stride=1, padding=0)

    def forward(self, f1, f2):
        b, c, h, w = f1.size()

        f1 = f1.reshape([b, c, -1])  # (b, c, h*w)
        f2 = f2.reshape([b, c, -1])  # (b, c, h*w)


        avg_1 = torch.mean(f1, dim=-1, keepdim=True).unsqueeze(-1)  # (b, c, 1, 1)
        max_1, _ = torch.max(f1, dim=-1, keepdim=True)  # (b, c, 1)
        max_1 = max_1.unsqueeze(-1)  # (b, c, 1, 1)

        avg_1 = F.relu(self.avg1(avg_1))
        max_1 = F.relu(self.max1(max_1))
        avg_1 = self.avg11(avg_1).squeeze(-1)  # (b, c, 1)
        max_1 = self.max11(max_1).squeeze(-1)  # (b, c, 1)
        a1 = avg_1 + max_1

        avg_2 = torch.mean(f2, dim=-1, keepdim=True).unsqueeze(-1)  # (b, c, 1, 1)
        max_2, _ = torch.max(f2, dim=-1, keepdim=True)  # (b, c, 1)
        max_2 = max_2.unsqueeze(-1)  # (b, c, 1, 1)

        avg_2 = F.relu(self.avg2(avg_2))
        max_2 = F.relu(self.max2(max_2))
        avg_2 = self.avg22(avg_2).squeeze(-1)  # (b, c, 1)
        max_2 = self.max22(max_2).squeeze(-1)  # (b, c, 1)
        a2 = avg_2 + max_2

        cross = torch.matmul(a1, a2.transpose(1, 2))  # (b, c, c)

        a1 = torch.matmul(F.softmax(cross, dim=-1), f1)  # (b, c, h*w)
        a2 = torch.matmul(F.softmax(cross.transpose(1, 2), dim=-1), f2)  # (b, c, h*w)

        a1 = a1.reshape([b, c, h, w])  # (b, c, h, w)
        avg_out = torch.mean(a1, dim=1, keepdim=True)  # (b, 1, h, w)
        max_out, _ = torch.max(a1, dim=1, keepdim=True)  # (b, 1, h, w)
        a1 = torch.cat([avg_out, max_out], dim=1)  # (b, 2, h, w)
        a1 = F.relu(self.conv1_spatial(a1))  # (b, 1, h, w)
        a1 = self.conv2_spatial(a1)  # (b, 1, h, w)
        a1 = a1.reshape([b, 1, -1])  # (b, 1, h*w)
        a1 = F.softmax(a1, dim=-1)  # (b, 1, h*w)

        a2 = a2.reshape([b, c, h, w])  # (b, c, h, w)
        avg_out = torch.mean(a2, dim=1, keepdim=True)  # (b, 1, h, w)
        max_out, _ = torch.max(a2, dim=1, keepdim=True)  # (b, 1, h, w)
        a2 = torch.cat([avg_out, max_out], dim=1)  # (b, 2, h, w)
        a2 = F.relu(self.conv1_spatial(a2))  # (b, 1, h, w)
        a2 = self.conv2_spatial(a2)  # (b, 1, h, w)
        a2 = a2.reshape([b, 1, -1])  # (b, 1, h*w)
        a2 = F.softmax(a2, dim=-1)  # (b, 1, h*w)


        f1 = f1 * a1 + f1  # (b, c, h*w)
        f2 = f2 * a2 + f2  # (b, c, h*w)

        f1 = f1.reshape([b, c, h, w])  # (b, c, h, w)
        f2 = f2.reshape([b, c, h, w])  # (b, c, h, w)

        return f1, f2


if __name__ == '__main__':
    block=DACF()
    f1=torch.randn([1,32,224,224])
    f2=torch.randn([1,32,224,224])
    x1,x2=block(f1,f2)
