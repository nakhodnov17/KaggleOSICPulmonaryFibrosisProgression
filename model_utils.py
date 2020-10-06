import tabulate

import numpy as np

import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler

import sys
sys.path.append('./../input/osicutils/')
sys.path.append('./../input/osicutils/my_efficientnet_pytorch_3d/')
from my_efficientnet_pytorch_3d import EfficientNet3D


class VGG(torch.nn.Module):
    _vgg_configurations = {
        'small': [8, 'M', 8, 'M', 16, 'M', 16, 'M'],
        8: [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'],
        11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }

    @staticmethod
    def _make_layers(cfg, batch_norm):
        layers = []
        in_channels = 1
        for v in cfg:
            if v == 'M':
                layers += [torch.nn.MaxPool3d(kernel_size=2, stride=2)]
            else:
                layers += [torch.nn.Conv3d(in_channels, v, kernel_size=3, padding=1)]
                if batch_norm:
                    layers += [torch.nn.BatchNorm3d(v)]
                layers += [torch.nn.ReLU(inplace=True)]
                in_channels = v
        return layers

    def __init__(self, VGG_version, batch_norm):
        super().__init__()
        self.VGG_version = VGG_version
        self.batch_norm = batch_norm

        self.net = nn.Sequential(
            *VGG._make_layers(self._vgg_configurations[VGG_version], batch_norm)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
            elif isinstance(m, torch.nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        return self.net(x)


class LaplaceLoss(nn.Module):  # _Loss):
    def forward(self, y_true, preds, log_sigma, metric=False):
        abs_diff = (y_true - preds).abs()

        log_sigma.clamp_(-np.log(70), np.log(70))

        if metric:
            abs_diff.clamp_max_(1000)

        losses = np.sqrt(2) * abs_diff / log_sigma.exp() + log_sigma + np.log(2) / 2
        return losses.mean()


class PinballLoss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = torch.tensor(alpha, dtype=torch.float32)

    def forward(self, y, z):
        return torch.max((y - z) * self.alpha, (z - y) * (1 - self.alpha)).mean()


class SqueezeLayer(nn.Module):
    def forward(self, x):
        return x.squeeze()


class FeatureExtractor(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net.extract_features(x)


class OSICNet(nn.Module):
    FVC_MEAN, FVC_STD = 2690.479018721756, 832.5021066817238

    def __init__(self, dtype, device, use_poly, use_quantiles, efficient_net_model_number, hidden_size,
                 dropout_rate):  # , output_size
        super().__init__()

        self.dtype = dtype
        self.device = device

        self.use_poly = use_poly
        self.use_quantiles = use_quantiles
        assert not (self.use_poly and self.use_quantiles)

        self.CT_features_extractor = nn.Sequential(
            # FeatureExtractor(
            #     EfficientNet3D.from_name(
            #         f'efficientnet-b{efficient_net_model_number}', override_params={'num_classes': 1}, in_channels=1
            #     )
            # ),
            VGG('small', True),
            nn.AdaptiveAvgPool3d(1),
            SqueezeLayer()
        )

        self.predictor = nn.Sequential(
            nn.Linear(16 + 15 + (1 - self.use_poly), hidden_size),  # 1280
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, (5 if self.use_poly else 2) + self.use_quantiles)
            # poly coefs & sigma or FVC & log_sigma
        )

        self._initialize_weights()

        self.CT_features_extractor.to(self.device)
        self.predictor.to(self.device)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
            elif isinstance(m, torch.nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _prepare_data(self, data):
        data = list(data)
        if data[0].ndim >= 2:
            for i in range(len(data)):
                data[i] = data[i].squeeze(0)

        percents, weeks, FVCs, features, masks, images = data

        lungs_mean, lungs_std = -971.4692260919278, 117.84143467421829
        lungs = -1000 * (1.0 - masks) + masks * images
        lungs = (lungs - lungs_mean) / lungs_std
        lungs = lungs.type(self.dtype)

        percents_mean, percents_std = 77.6726, 19.8233
        weeks_mean, weeks_std = 31.861846352485475, 23.240045178171002

        percents = (percents - percents_mean) / percents_std
        weeks = (weeks - weeks_mean) / weeks_std
        FVCs = (FVCs - self.FVC_MEAN) / self.FVC_STD
        features = features.type(self.dtype)

        return percents, weeks, FVCs, features, lungs, images

    def forward(self, data):
        weeks_unnorm = data[1]
        percents, weeks, FVCs, features, lungs, images = self._prepare_data(data)

        lungs = lungs.unsqueeze(0).to(self.device)
        lungs_features = self.CT_features_extractor(lungs)

        all_preds = []
        for base_percent, base_week, base_FVC in zip(percents, weeks, FVCs):
            table_features = torch.cat([
                torch.tensor([base_percent]),
                torch.tensor([base_week]),
                torch.tensor([base_FVC]),
                features
            ]).to(self.device)

            all_features = torch.cat([lungs_features, table_features])

            if self.use_poly:
                X = all_features
            else:
                X = torch.cat([all_features.repeat(weeks.shape[0], 1), weeks.unsqueeze(1).to(self.device)], dim=1)

            preds = self.predictor(X).cpu()

            if self.use_poly:
                weeks_poly = torch.empty(len(weeks), 4, dtype=self.dtype)
                weeks_poly[:, 0] = weeks_unnorm ** 3
                weeks_poly[:, 1] = weeks_unnorm ** 2
                weeks_poly[:, 2] = weeks_unnorm
                weeks_poly[:, 3] = 1

                coefs = preds[:4]
                log_sigmas = preds[4]

                FVC_preds = (weeks_poly * coefs).sum(dim=1)
                log_sigmas = log_sigmas.repeat(FVC_preds.shape[0])
            else:
                if self.use_quantiles:
                    FVC_low, FVC_preds, FVC_high = preds.transpose(0, 1) * self.FVC_STD + self.FVC_MEAN
                    all_preds.append((FVC_low, FVC_preds, FVC_high))
                    continue
                FVC_preds, log_sigmas = preds.transpose(0, 1)

            FVC_preds = FVC_preds * self.FVC_STD + self.FVC_MEAN

            all_preds.append((FVC_preds, log_sigmas))
        return all_preds


class LinearDecayLR(_LRScheduler):
    def __init__(self, optimizer, start_epoch, stop_epoch, start_lr, stop_lr, last_epoch=-1):
        self.optimizer = optimizer

        self.start_epoch = start_epoch
        self.stop_epoch = stop_epoch

        self.start_lr = start_lr
        self.stop_lr = stop_lr

        self.last_epoch = last_epoch

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list:
        if self.last_epoch < self.start_epoch:
            new_lr = self.start_lr
        elif self.last_epoch > self.stop_epoch:
            new_lr = self.stop_lr
        else:
            new_lr = self.start_lr + (
                    (self.stop_lr - self.start_lr) *
                    (self.last_epoch - self.start_epoch) /
                    (self.stop_epoch - self.start_epoch)
            )
        return [new_lr for _ in self.optimizer.param_groups]


def print_results(mode, writer, iter_num, metrics, log_sigmas):
    for metric, values in metrics.items():
        value = values[-1]
        writer.add_scalar(f'{metric}/{mode}', value, iter_num)
    writer.add_scalar(f'sigma/{mode}', np.exp(log_sigmas.detach().mean().item()), iter_num)

    columns = [
        'Iter',
    ]

    values = [
        f'{iter_num}',
    ]

    for metric, cur_values in metrics.items():  # sorted(metrics.items(), key=lambda x: x[0]):
        columns.append(metric.replace('laplace_loss', 'll'))
        values.append(cur_values[-1])

    columns += ['Sigma']
    values += [np.exp(log_sigmas.detach().mean().item())]

    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    if iter_num % 40 == 1:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]

    print(table)