
def get_fusion_model(model_name, num_class=2):
    if  model_name == 'MobileNetV2':
        from model.mobilenet import MobileNetV2_dual
        net = MobileNetV2_dual(num_classes=num_class)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return net

def get_model(model_name, num_class=2):
    if model_name == 'MobileNetV2':
        from model.mobilenet import MobileNetV2_single
        net = MobileNetV2_single(num_classes=num_class)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return net