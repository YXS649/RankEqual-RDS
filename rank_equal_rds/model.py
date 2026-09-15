"""Shared MLP and fourteen source heads; no alternative architectures."""
import torch.nn as nn
import torch.nn.functional as F


class CFE(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = nn.Sequential(
            nn.Linear(310, 256), nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(256, 128), nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(128, 64), nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.module(x)


class DSFE(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = nn.Sequential(
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.module(x)


class RankEqualNet(nn.Module):
    def __init__(self, number_of_source=14, number_of_category=3):
        super().__init__()
        if number_of_source != 14 or number_of_category not in (3, 4):
            raise ValueError('The main method requires 14 sources and 3 or 4 classes')
        self.sharedNet = CFE()
        self.number_of_source = number_of_source
        self.number_of_category = number_of_category
        # Preserve initialization order and state-dict names of the original model.
        for i in range(number_of_source):
            self.add_module(f'DSFE{i}', DSFE())
            self.add_module(f'cls_fc_DSC{i}', nn.Linear(32, number_of_category))

    def forward(self, data, number_of_source=None):
        if number_of_source not in (None, self.number_of_source):
            raise ValueError('Inference traverses all fourteen heads')
        z = self.sharedNet(data)
        return [getattr(self, f'cls_fc_DSC{i}')(getattr(self, f'DSFE{i}')(z))
                for i in range(self.number_of_source)]

    def training_loss_inputs(self, source, labels, target, head):
        """Keep target-through-all-heads BN behavior, including during warm-up.

        Only the current source CE and current head's target MCC enter the loss.
        Inactive participation is not a promise of frozen BatchNorm buffers.
        """
        source_z = self.sharedNet(source)
        target_z = self.sharedNet(target)
        target_features = [getattr(self, f'DSFE{i}')(target_z) for i in range(14)]
        target_logits = [getattr(self, f'cls_fc_DSC{i}')(target_features[i]) for i in range(14)]
        source_logits = getattr(self, f'cls_fc_DSC{head}')(
            getattr(self, f'DSFE{head}')(source_z))
        return F.cross_entropy(source_logits, labels.view(-1).long()), target_logits[head]
