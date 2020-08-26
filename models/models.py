"""
Aus der originalen Implementation
Enthält die Funktion create_model
"""


def create_model(opt):
    """ Initilaisiert das Netzwerk über die Klasse UNetModel """
    model = None
    print(opt.model)
    if opt.model == 'unet':
        from .unet_model import UNetModel
        model = UNetModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
