import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)

class DN3BaseModel(nn.Module):
    def __init__(self, samples, channels, return_features=True):
        super().__init__()
        self.samples = samples
        self.channels = channels
        self.return_features = return_features

    def forward(self, x):
        raise NotImplementedError

    def internal_loss(self, forward_pass_tensors):
        return None

    def clone(self):
        return deepcopy(self)

    def load(self, filename, strict=True):
        state_dict = torch.load(filename)
        self.load_state_dict(state_dict, strict=strict)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def freeze_features(self, unfreeze=False):
        for param in self.parameters():
            param.requires_grad = unfreeze


class Classifier(DN3BaseModel):
    def __init__(self, targets, samples, channels, return_features=False):
        super(Classifier, self).__init__(samples, channels, return_features=return_features)
        self.targets = targets
        self.make_new_classification_layer()
        self._init_state = self.state_dict()

    def reset(self):
        self.load_state_dict(self._init_state)

    def forward(self, X1, X2, padding_masks):
        features = self.features_forward(X1, X2, padding_masks)
        if self.return_features:
            return self.classifier_forward(features), features
        else:
            return self.classifier_forward(features)

    def make_new_classification_layer(self):
        classifier = nn.Linear(self.num_features_for_classification, self.targets)
        nn.init.xavier_normal_(classifier.weight)
        classifier.bias.data.zero_()
        self.classifier = nn.Sequential(Flatten(), classifier)

    def freeze_features(self, unfreeze=False, freeze_classifier=False):
        super(Classifier, self).freeze_features(unfreeze=unfreeze)
        if isinstance(self.classifier, nn.Module) and not freeze_classifier:
            for param in self.classifier.parameters():
                param.requires_grad = True

    @property
    def num_features_for_classification(self):
        raise NotImplementedError

    def classifier_forward(self, features):
        return self.classifier(features)

    def features_forward(self, x):
        raise NotImplementedError

    def load(self, filename, include_classifier=False, freeze_features=True):
        state_dict = torch.load(filename)
        if not include_classifier:
            for key in [k for k in state_dict.keys() if 'classifier' in k]:
                state_dict.pop(key)
        self.load_state_dict(state_dict, strict=False)
        if freeze_features:
            self.freeze_features()

    def save(self, filename, ignore_classifier=False):
        state_dict = self.state_dict()
        if ignore_classifier:
            for key in [k for k in state_dict.keys() if 'classifier' in k]:
                state_dict.pop(key)
        print("Saving to {} ...".format(filename))
        torch.save(state_dict, filename)