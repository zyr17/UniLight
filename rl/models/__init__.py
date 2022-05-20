import os

from . import wrapperModels
from . import innerModels

output_models = []

def parseclass(module, baseclass):
    vm = vars(module)
    for key in vm:
        try:
            if issubclass(vm[key], baseclass):
                globals()[key] = vm[key]
                globals()['output_models'].append(key)
                # print('add', key)
            else:
                # print('reject', key)
                pass
        except Exception:
            # print('except', key)
            pass

parseclass(wrapperModels, wrapperModels.WrapperModelBase)
parseclass(innerModels, innerModels.InnerModelBase)

__all__ = output_models
