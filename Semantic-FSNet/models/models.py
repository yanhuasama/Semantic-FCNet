import torch


models = {}
def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def make(name, args = None, **kwargs):
    if name is None:
        return None
    if args is not None:
        model = models[name](args, **kwargs)
    else:
        model = models[name](**kwargs)

    if torch.cuda.is_available():
        model.cuda()
    return model


def load(model_sv, name=None, args=None):
    if name is None:
        name = 'model'
    model = make(model_sv[name], args, **model_sv[name + '_args'])
    model.load_state_dict(model_sv[name + '_sd'])
    return model

