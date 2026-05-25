import copy


models = {}

# 将模型类注册到全局models字典中，可以通过名称动态地创建模型实例
# 使用方法：@register('name')
def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator

# 根据模型相关超参数创建和初始化模型实例
def make(model_spec, args=None, load_sd=False):
    if args is not None:
        model_args = copy.deepcopy(model_spec['args'])
        model_args.update(args)
    else:
        model_args = model_spec['args']

    #TODO 根据配置文件中name创建模型实例
    model = models[model_spec['name']](**model_args)
    
    if load_sd: # 加载模型状态
        model.load_state_dict(model_spec['sd'])
    return model
