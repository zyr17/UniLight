"""import all classes in the folder. for file `X.py`, the only the class `X`
   defined in the file will be imported.
"""
import os
import importlib

innermodel_folder = os.path.dirname(os.path.realpath(__file__))
innerModels = os.listdir(innermodel_folder)

output_models = []

for innerModel in innerModels:
    if innerModel[-3:] == '.py' and innerModel != '__init__.py':
        innerModel = innerModel[:-3]
        im = importlib.import_module(__package__ + '.' + innerModel)

        globals()[innerModel] = vars(im)[innerModel]
        output_models.append(innerModel)

__all__ = output_models
