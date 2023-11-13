import torch
import torch.nn as nn
from typing import List, Optional

class Configs():
    n_blocks: List[int] = [1, 2, 3, 3, 3, 3, 0, 2]
    n_channels: List[int] = [16, 32, 48, 64, 128, 256, 256, 300]
    context_len: int = 4

    bottlenecks: Optional[List[int]] = [16, 32, 48, 64, 128, 256, 256, 300]
    first_kernel_size: int = 3
    device = 'cuda'


class PositionalEncoding(nn.Module): # Probably unnecessary. Basically just a wrapper around the embedding layer
    def __init__(self, num_positions, embedding_dim):
        super().__init__()
        self.positional_encoding = nn.Embedding(num_positions, embedding_dim)

    def forward(self, position_ids):
        # Retrieves the positional encoding for each position id
        return self.positional_encoding(position_ids)


class ShortcutBlock(nn.Module): # Standard shortcut block
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.batchnorm(self.conv(x))


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # If it is not an in-place transform, make sure the shortcut matches the dimensionality
        # of the ResBlock. If the dimensionality is the same, just use the identity of the input
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ShortcutBlock(in_channels, out_channels, stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        return torch.relu(x + shortcut)


class BottleneckResBlock(nn.Module):  # Standard bottleneck. Reduces the complexity of the ResNet blocks
    def __init__(self, in_channels, bottleneck_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)

        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)

        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = ShortcutBlock(in_channels, out_channels, stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv3(x)))
        x = self.bn3(self.conv3(x))

        return torch.relu(x + shortcut)


class ResNet(nn.Module):
    def __init__(self, n_blocks, n_channels, bottlenecks, img_channels=3, first_kernel_size=7) -> None:
        super().__init__()
        assert len(n_blocks) == len(n_channels)
        assert bottlenecks is None or len(bottlenecks) == len(n_channels)

        self.conv = nn.Conv2d(img_channels, n_channels[0], kernel_size=first_kernel_size, stride=1, padding=first_kernel_size//2)
        self.bn = nn.BatchNorm2d(n_channels[0])

        blocks = []
        prev_channels = n_channels[0]
        for i, channels in enumerate(n_channels):
            # First layer reduce dimensionality (ugly to do this but idc)
            stride = 2 if len(blocks) == 0 else 1
            if bottlenecks is None:
                blocks.append(ResBlock(prev_channels, channels, stride=stride))
            else:
                blocks.append(BottleneckResBlock(prev_channels, bottlenecks[i], channels, stride=1))

            # Every other layer group (NOT EVERY LAYER, EVERY GROUP) reduce the dimensionality by 2 w/ stride
            if i % 2 == 1:
              blocks.append(nn.Conv2d(channels, channels, kernel_size=first_kernel_size, stride=2, padding=first_kernel_size//2))
            prev_channels = channels
            for _ in range(n_blocks[i] - 1):
                if bottlenecks is None:
                    blocks.append(ResBlock(channels, channels, stride=1))
                else:
                    blocks.append(BottleneckResBlock(channels, bottlenecks[i], channels, stride=1))


            self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.bn(self.conv(x))
        x = self.blocks(x)
        x = x.view(x.shape[0], -1)

        return x


class Model(nn.Module):
    def __init__(self, c: Configs, num_positions=55, embedding_dim=10):
        super().__init__()
        self.emb_dim = embedding_dim
        self.c = c

        # RESIDUAL MODEL
        self.base = ResNet(c.n_blocks, c.n_channels, c.bottlenecks, img_channels=c.context_len, first_kernel_size=c.first_kernel_size)
        self.nl = nn.Tanh()

        # Calculate input size for FFNN. Ugly right now. #TODO: make not ugly!
        input_sz = (300*4*4) + self.emb_dim*c.context_len + self.emb_dim*c.context_len + 50
        self.fc1 = nn.Linear(input_sz, 2000)
        self.fc2 = nn.Linear(2000, 1000)
        self.fc3 = nn.Linear(1000, 64*64)

        self.context_positional_encoding = PositionalEncoding(num_positions, embedding_dim)
        self.time_encoding = PositionalEncoding(20, embedding_dim)
        self.target_encoding = PositionalEncoding(num_positions, 50)

    def forward(self, x, positions, times, target):
        x = self.base(x)
        x = x.view(x.shape[0], -1)
        x = self.nl(x)
        position_embeddings = self.context_positional_encoding(positions)
        position_embeddings = position_embeddings.view(-1, self.emb_dim*self.c.context_len)

        time_embeddings = self.time_encoding(times + 10)
        time_embeddings = time_embeddings.view(-1, self.emb_dim*self.c.context_len)

        target_embedding = self.target_encoding(target)

        input_with_position = torch.cat((x, position_embeddings, time_embeddings, target_embedding), dim=-1)

        x = self.nl(self.fc1(input_with_position))
        x = self.nl(self.fc2(x))
        x = self.fc3(x)
        return x
