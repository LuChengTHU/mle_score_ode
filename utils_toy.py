import torch

class ExponentialMovingAverage(object):

    def __init__(self, module, decay=0.999):
        """Initializes the model when .apply() is called the first time.
        This is to take into account data-dependent initialization that occurs in the first iteration."""
        self.module = module
        self.decay = decay
        self.shadow_params = {}
        self.nparams = sum(p.numel() for p in module.parameters())

    def init(self):
        for name, param in self.module.named_parameters():
            self.shadow_params[name] = param.data.clone()

    def apply(self):
        if len(self.shadow_params) == 0:
            self.init()
        else:
            with torch.no_grad():
                for name, param in self.module.named_parameters():
                    self.shadow_params[name] -= (1 - self.decay) * (self.shadow_params[name] - param.data)

    def set(self, other_ema):
        self.init()
        with torch.no_grad():
            for name, param in other_ema.shadow_params.items():
                self.shadow_params[name].copy_(param)

    def replace_with_ema(self):
        for name, param in self.module.named_parameters():
            param.data.copy_(self.shadow_params[name])

    def swap(self):
        for name, param in self.module.named_parameters():
            tmp = self.shadow_params[name].clone()
            self.shadow_params[name].copy_(param.data)
            param.data.copy_(tmp)

    def __repr__(self):
        return (
            '{}(decay={}, module={}, nparams={})'.format(
                self.__class__.__name__, self.decay, self.module.__class__.__name__, self.nparams
            )
        )