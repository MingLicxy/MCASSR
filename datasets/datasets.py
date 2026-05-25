import copy


datasets = {}

# 将数据类注册到全局datasets字典中，可以通过名称动态地创建数据集实例
# 使用方法：@register('name')
def register(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator

# 根据数据集相关超参数创建数据集实例
def make(dataset_spec, args=None):
    if args is not None:
        dataset_args = copy.deepcopy(dataset_spec['args'])
        dataset_args.update(args)
    else:
        dataset_args = dataset_spec['args']
    
    #TODO 根据配置文件中name创建数据集实例
    dataset = datasets[dataset_spec['name']](**dataset_args)
    return dataset
