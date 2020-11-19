# from mini_imagenet_dataloader import MiniImageNetDataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np
from torchvision import transforms
from torchmeta.transforms import ClassSplitter, Categorical
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.datasets.miniimagenet import MiniImagenet
from collections import OrderedDict
import torch.nn.functional as F
from torchmeta.modules import (MetaModule, MetaSequential, MetaConv2d,
                               MetaBatchNorm2d, MetaLinear)
from torchmeta.utils.gradient_based import gradient_update_parameters

def conv_block(in_channels, out_channels, **kwargs):
    return MetaSequential(OrderedDict([
        ('conv', MetaConv2d(in_channels, out_channels, **kwargs)),
        ('norm', nn.BatchNorm2d(out_channels, momentum=1.,
            track_running_stats=False)),
        ('relu', nn.ReLU()),
        ('pool', nn.MaxPool2d(2))
    ]))


def get_accuracy(logits, targets):
    """Compute the accuracy (after adaptation) of MAML on the test/query points
    Parameters
    ----------
    logits : `torch.FloatTensor` instance
        Outputs/logits of the model on the query points. This tensor has shape
        `(num_examples, num_classes)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has 
        shape `(num_examples,)`.
    Returns
    -------
    accuracy : `torch.FloatTensor` instance
        Mean accuracy on the query points
    """
    _, predictions = torch.max(logits, dim=-1)
    return torch.mean(predictions.eq(targets).float())

class MetaConvModel(MetaModule):
    """4-layer Convolutional Neural Network architecture from [1].
    Parameters
    ----------
    in_channels : int
        Number of channels for the input images.
    out_features : int
        Number of classes (output of the model).
    hidden_size : int (default: 64)
        Number of channels in the intermediate representations.
    feature_size : int (default: 64)
        Number of features returned by the convolutional head.
    References
    ----------
    .. [1] Finn C., Abbeel P., and Levine, S. (2017). Model-Agnostic Meta-Learning
           for Fast Adaptation of Deep Networks. International Conference on
           Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)
    """
    def __init__(self, in_channels, out_features, hidden_size=64, feature_size=64):
        super(MetaConvModel, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size
        self.feature_size = feature_size

        self.features = MetaSequential(OrderedDict([
            ('layer1', conv_block(in_channels, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True)),
            ('layer2', conv_block(hidden_size, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True)),
            ('layer3', conv_block(hidden_size, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True)),
            ('layer4', conv_block(hidden_size, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True))
        ]))
        self.classifier = MetaLinear(feature_size, out_features, bias=True)
        
    def forward(self, inputs, params=None):
        features = self.features(inputs, params=self.get_subdict(params, 'features'))
        features = features.view((features.size(0), -1))
        logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))
        return logits


def ModelConvMiniImagenet(out_features, hidden_size=84):
    return MetaConvModel(3, out_features, hidden_size=hidden_size,
                         feature_size=5 * 5 * hidden_size)
def train():
    # dataloader = MiniImageNetDataLoader(shot_num=5, way_num=5, episode_test_sample_num=15)

    # dataloader.load_list(phase='all')

    # episode_train_img, episode_train_label, episode_test_img, episode_test_label = \
    #     dataloader.get_batch(phase='train', idx=0)

    transform = transforms.Compose([
        # you can add other transformations in this list
        transforms.Resize(84),
        transforms.ToTensor()
    ])
    dataset_transform = ClassSplitter(shuffle=True, num_train_per_class=5, num_test_per_class=5)
    dataset = MiniImagenet('', transform=transform, num_classes_per_task=5, target_transform=Categorical(num_classes=5) ,meta_split="train", dataset_transform=dataset_transform )

    dataloader = BatchMetaDataLoader(dataset, batch_size=1, shuffle=True)
    
    model = ModelConvMiniImagenet(5)
    model.to(device='cpu')
    model.train()
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    with tqdm(dataloader, total=25) as pbar:
        for batch_idx, batch in enumerate(pbar):

            model.zero_grad()

            train_inputs, train_targets = batch['train']
            train_inputs = train_inputs.to(device='cpu')
            train_targets = train_targets.to(device='cpu')

            test_inputs, test_targets = batch['test']
            test_inputs = test_inputs.to(device='cpu')
            test_targets = test_targets.to(device='cpu')

            outer_loss = torch.tensor(0., device='cpu')
            accuracy = torch.tensor(0., device='cpu')
            for task_idx, (train_input, train_target, test_input,
                    test_target) in enumerate(zip(train_inputs, train_targets,
                    test_inputs, test_targets)):
               
                train_logit = model(train_input)
                inner_loss = F.cross_entropy(train_logit, train_target)

                model.zero_grad()
                params = gradient_update_parameters(model, inner_loss)

                test_logit = model(test_input, params=params)
                outer_loss += F.cross_entropy(test_logit, test_target)

                with torch.no_grad():
                    accuracy += get_accuracy(test_logit, test_target)
            outer_loss.div_(1)
            accuracy.div_(1)

            outer_loss.backward()
            meta_optimizer.step()

            pbar.set_postfix(accuracy='{0:.4f}'.format(accuracy.item()))
            if(batch_idx >= 25):
                break

if __name__ == "__main__":
    train()