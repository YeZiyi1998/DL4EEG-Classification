from model.bendr.models import Classifier
from model.bendr.layers import ConvEncoderBENDR, BENDRContextualizer
import tqdm
import torch.nn as nn

class BENDRClassification(Classifier):
    @property
    def num_features_for_classification(self):
        return self.encoder_h

    def features_forward(self, X1, X2, padding_masks):
        encoded = self.encoder(X2)

        if self.trial_embeddings is not None and len(x) > 1:
            embeddings = self.trial_embeddings(x[-1])
            encoded += embeddings.unsqueeze(-1).expand_as(encoded)

        context = self.contextualizer(encoded)

        return context[:, :, -1]

    def __init__(self, args, input_dim, num_nodes, device,  targets = 2, encoder_h = 512, contextualizer_hidden = 3076, projection_head=False,
                 new_projection_layers=0, dropout=0., trial_embeddings=None, layer_drop=0, keep_layers=None,
                 mask_p_t = 0.01, mask_p_c = 0.005, mask_t_span=0.1, mask_c_span=0.1, multi_gpu=False):
        samples = input_dim
        channels = num_nodes
        self.encoder_h = encoder_h
        self.contextualizer_hidden = contextualizer_hidden
        super().__init__(targets, samples, channels)

        encoder = ConvEncoderBENDR(channels, encoder_h=encoder_h, dropout=dropout, projection_head=projection_head)
        encoded_samples = encoder.downsampling_factor(samples)
        encoder.load('models/encoder.pt')

        mask_t_span = mask_t_span if mask_t_span > 1 else int(mask_t_span * encoded_samples)
        mask_c_span = mask_c_span if mask_c_span > 1 else int(mask_c_span * encoder_h)
        contextualizer = BENDRContextualizer(encoder_h, hidden_feedforward=contextualizer_hidden, finetuning=True, mask_p_t=mask_p_t, mask_p_c=mask_p_c, layer_drop=layer_drop, mask_c_span=mask_c_span, dropout=dropout,mask_t_span=mask_t_span)
        contextualizer.load('models/contextualizer.pt')
        self.encoder = nn.DataParallel(encoder) if multi_gpu else encoder
        self.contextualizer = nn.DataParallel(contextualizer) if multi_gpu else contextualizer

        tqdm.tqdm.write(encoder.description(sequence_len=samples))

        self.projection_mlp = nn.Sequential()
        for p in range(1, new_projection_layers + 1):
            self.projection_mlp.add_module("projection-{}".format(p), nn.Sequential(
                nn.Linear(encoder_h, encoder_h),
                nn.Dropout(dropout),
                nn.BatchNorm1d(encoder_h),
                nn.GELU(),
            ))
        self.trial_embeddings = nn.Embedding(trial_embeddings, encoder_h, scale_grad_by_freq=True) \
            if trial_embeddings is not None else trial_embeddings

    def load_encoder(self, encoder_file, freeze=False, strict=True):
        self.encoder.load(encoder_file, strict=strict)
        self.encoder.freeze_features(unfreeze=not freeze)

    def load_contextualizer(self, contextualizer_file, freeze=False, strict=True):
        self.contextualizer.load(contextualizer_file, strict=strict)
        self.contextualizer.freeze_features(unfreeze=not freeze)

    def load_pretrained_modules(self, encoder_file, contextualizer_file, freeze_encoder=False,
                                freeze_contextualizer=False, freeze_position_conv=False,
                                freeze_mask_replacement=True, strict=False):
        self.load_encoder(encoder_file, freeze=freeze_encoder, strict=strict)
        self.load_contextualizer(contextualizer_file, freeze=freeze_contextualizer, strict=strict)
        self.contextualizer.mask_replacement.requires_grad = freeze_mask_replacement
        if freeze_position_conv:
            for p in self.contextualizer.relative_position.parameters():
                p.requires_grad = False


