import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models import register
from einops import rearrange

################################ 实现 Fourier Encoding 模块 ##################################
class FourierEmbedding(nn.Module):
    def __init__(self, in_dim, num_frequencies, include_input=True, log_sampling=True):
        super().__init__()
        self.in_dim = in_dim
        self.num_frequencies = num_frequencies
        self.include_input = include_input

        if log_sampling:
            self.freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        else:
            self.freq_bands = torch.linspace(1.0, 2.0 ** (num_frequencies - 1), num_frequencies)

    def forward(self, x):
        # x: [..., in_dim]
        out = []
        if self.include_input:
            out.append(x)

        for freq in self.freq_bands.to(x.device):
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))

        return torch.cat(out, dim=-1)

    @property
    def out_dim(self):
        dim = 0
        if self.include_input:
            dim += self.in_dim
        dim += 2 * self.in_dim * self.num_frequencies
        return dim

################################ 傅里叶频域解码器 ##################################
@register('fourier_mlp')
class FourierMLP(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 hidden_list,
                 num_frequencies=10,
                 include_input=True,
                 act='relu',
                 final_act=False,
                 act_trainable=False,
                 **kwargs):
        super().__init__()

        # Fourier Encoding
        self.embedder = FourierEmbedding(
            in_dim=in_dim,
            num_frequencies=num_frequencies,
            include_input=include_input
        )
        mlp_in_dim = self.embedder.out_dim

        # ===== 复用你原来的 MLP 逻辑 =====
        if act is None:
            self.act = None
        elif act.lower() == 'relu':
            self.act = nn.ReLU()
        elif act.lower() == 'gelu':
            self.act = nn.GELU()
        elif act.lower() == 'sine':
            self.act = Siren()
        elif act.lower() == 'expsin':
            self.act = ExpSinActivation(a=kwargs['a'], trainable=act_trainable)
        elif act.lower() == 'gaussian':
            self.act = GaussianActivation(a=kwargs['a'], trainable=act_trainable)
        elif act.lower() == 'quadratic':
            self.act = QuadraticActivation(a=kwargs['a'], trainable=act_trainable)
        elif act.lower() == 'multi_quadratic':
            self.act = MultiQuadraticActivation(a=kwargs['a'], trainable=act_trainable)
        elif act.lower() == 'laplacian':
            self.act = LaplacianActivation(a=kwargs['a'], trainable=act_trainable)
        elif act.lower() == 'super_gaussian':
            self.act = SuperGaussianActivation(a=kwargs['a'], b=kwargs['b'], trainable=act_trainable)
        else:
            raise NotImplementedError(f'activation {act} not supported')

        layers = []
        lastv = mlp_in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            if self.act:
                layers.append(self.act)
            lastv = hidden

        layers.append(nn.Linear(lastv, out_dim))
        if final_act and self.act:
            layers.append(self.act)

        self.layers = nn.Sequential(*layers)

        # SIREN initialization
        if act is not None and act.lower() == 'sine':
            self.layers.apply(sine_init)
            self.layers[0].apply(first_layer_sine_init)

    def forward(self, x):
        # x: [..., in_dim]
        shape = x.shape[:-1]
        x = self.embedder(x)
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)





################################ MLP解码器 ##################################
@register('mlp')
class MLP(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 hidden_list,
                 act='relu',
                 final_act=False,
                 act_trainable=False, #TODO 激活函数超参是否可训练
                  **kwargs): #TODO 定义激活函数超参'a'与'b'
        super().__init__()
        
        # 选择MLP中的激活函数
        if act is None:
            self.act = None
        elif act.lower() == 'relu':
            self.act = nn.ReLU() 
        elif act.lower() == 'gelu':
            self.act = nn.GELU()
        elif act.lower() == 'sine':
            self.act = Siren()
        elif act.lower() == 'expsin':
            self.act = ExpSinActivation(a=kwargs['a'], trainable=act_trainable)
        elif act.lower() == 'gaussian':
            self.act = GaussianActivation(a=kwargs['a'], trainable=act_trainable)
        elif act.lower() == 'quadratic':
            self.act = QuadraticActivation(a=kwargs['a'], trainable=act_trainable)
        elif act.lower() == 'multi_quadratic':
            self.act = MultiQuadraticActivation(a=kwargs['a'], trainable=act_trainable)
        elif act.lower() == 'laplacian':
            self.act = LaplacianActivation(a=kwargs['a'], trainable=act_trainable) 
        elif act.lower() == 'super_gaussian':
            self.act = SuperGaussianActivation(a=kwargs['a'], b=kwargs['b'], trainable=act_trainable)  
        else:
            assert False, f'activation {act} is not supported'

        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            # 依据act确定MLP中的激活函数
            if self.act:
                layers.append(self.act)
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        
        #TODO final_act决定输出层后是否添加激活函数
        if final_act == True:
            if self.act:
                layers.append(self.act)
        self.layers = nn.Sequential(*layers)

        # 以正弦函数（sine）作为激活函数的神经网络需要特定的初始化方法
        if act is not None and act.lower() == 'sine':
            self.layers.apply(sine_init)
            self.layers[0].apply(first_layer_sine_init)

    def forward(self, x):
        #print('#####################################', x.shape) # [18432, 31]
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)

################################ ResMLP解码器 ##################################

# 启用异常检测
#torch.autograd.set_detect_anomaly(True)

@register('resmlp')
class ResMLP(nn.Module):
    
    #TODO 关键是根据hidden_list计算出所需ResBlock的数量
    def __init__(self,
                 in_dim,
                 out_dim,
                 hidden_list,
                 act='relu',
                 act_trainable=False, #TODO 激活函数超参是否可训练
                 block_type='relu_start',
                  **kwargs): #TODO 定义激活函数超参'a'与'b'
        super().__init__()

        # 选择MLP中的激活函数
        if act is None:
            self.act = None
        elif act.lower() == 'relu':
            self.act = nn.ReLU() 
        elif act.lower() == 'gelu':
            self.act = nn.GELU()
        elif act.lower() == 'sine':
            self.act = Siren()
        elif act.lower() == 'expsin':
            self.act = ExpSinActivation(a=kwargs['a'], trainable=act_trainable)
        elif act.lower() == 'gaussian':
            self.act = GaussianActivation(a=kwargs['a'], trainable=act_trainable)
        elif act.lower() == 'quadratic':
            self.act = QuadraticActivation(a=kwargs['a'], trainable=act_trainable)
        elif act.lower() == 'multi_quadratic':
            self.act = MultiQuadraticActivation(a=kwargs['a'], trainable=act_trainable)
        elif act.lower() == 'laplacian':
            self.act = LaplacianActivation(a=kwargs['a'], trainable=act_trainable) 
        elif act.lower() == 'super_gaussian':
            self.act = SuperGaussianActivation(a=kwargs['a'], b=kwargs['b'], trainable=act_trainable)  
        else:
            assert False, f'activation {act} is not supported'

        ################### 并未只在隐藏层实现残差块（残差块可能会包含输入/输出层）##################
        layers = []
        #TODO hidden_list=[256, 256]对应两个ResBlockReLUStart块
        if block_type == 'relu_start':
             lastv = hidden_list[0]
             layers.append(nn.Linear(in_dim, lastv)) # 输入层
             for hidden in hidden_list[1:]:
                 layers.append(ResBlockReLUStart(lastv, hidden, self.act))
                 lastv = hidden
             layers.append(ResBlockReLUStart(lastv, out_dim, self.act))

        #TODO hidden_list=[256, 256]对应两个ResBlockLinearStart块
        elif block_type == 'linear_start':
             lastv = in_dim
             for hidden in hidden_list:
                 layers.append(ResBlockLinearStart(lastv, hidden, self.act))
                 lastv = hidden  
             layers.append(nn.Linear(lastv, out_dim)) # 输出层
        self.layers = nn.Sequential(*layers)

         # 以正弦函数（sine）作为激活函数的神经网络需要特定的初始化方法
        if act is not None and act.lower() == 'sine':
            self.layers.apply(sine_init)
            self.layers[0].apply(first_layer_sine_init)
            
        #TODO
        #assert in_dim == out_dim

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)

# [ReLU, Linear, ReLU, Linear]
class ResBlockReLUStart(nn.Module):
    def __init__(self, in_dim, out_dim, act): # act传入激活函数
        super().__init__()
        self.act1 = act
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.act2 = act
        self.linear2 = nn.Linear(out_dim, out_dim)

        if in_dim != out_dim: # 防止残差连接维度对不上
            self.residual_layer = nn.Linear(in_dim, out_dim)
        else:
            self.residual_layer = None
        
        # 残差块只包含隐藏层时in_dim=out_dim
        #assert in_dim == out_dim

    def forward(self, x):
        if self.residual_layer:
            residual = self.residual_layer(x)
        else:
            residual = x
        out = self.act1(x)
        out = self.linear1(out)
        out = self.act2(out)
        out = self.linear2(out)
        out += residual
        return out

# [Linear, ReLU, Linear, ReLU]
class ResBlockLinearStart(nn.Module):
    def __init__(self, in_dim, out_dim, act):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.act1 = act
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.act2 = act

        if in_dim != out_dim:
            self.residual_layer = nn.Linear(in_dim, out_dim)
        else:
            self.residual_layer = None
        
        #assert in_dim == out_dim

    #TODO relu()的重新实现代替nn.ReLU()
    def custom_relu(self, x):
        return torch.maximum(x, torch.zeros_like(x))

    def forward(self, x):

        if self.residual_layer:
            residual = self.residual_layer(x)
        else:
            residual = x
        out = self.linear1(x)
        #TODO 尝试采用F.relu()代替nn.ReLU()
        #out = self.act1(out)
        #out = F.relu(out, inplace=False)
        out = self.custom_relu(out)
        out = self.linear2(out)
        #out = self.act2(out)
        #out = F.relu(out, inplace=False)
        out = self.custom_relu(out)
        out += residual
        return out


#TODO 各种简单激活函数
class GaussianActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return torch.exp(-x**2/(2*self.a**2))


class QuadraticActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return 1/(1+(self.a*x)**2)


class MultiQuadraticActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return 1/(1+(self.a*x)**2)**0.5


class LaplacianActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return torch.exp(-torch.abs(x)/self.a)


class SuperGaussianActivation(nn.Module):
    def __init__(self, a=1., b=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))
        self.register_parameter('b', nn.Parameter(b*torch.ones(1), trainable))

    def forward(self, x):
        return torch.exp(-x**2/(2*self.a**2))**self.b


class ExpSinActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return torch.exp(-torch.sin(self.a*x))


#TODO SIREN正弦激活函数
class Siren(nn.Module):
    """
        Siren activation
        https://arxiv.org/abs/2006.09661
    """

    def __init__(self, w0=30):
        """
            w0 comes from the end of section 3
            it should be 30 for the first layer
            and 1 for the rest
        """
        super().__init__()
        self.w0 = torch.tensor(w0)

    def forward(self, x):
        return torch.sin(self.w0 * x)

    def extra_repr(self):
        return "w0={}".format(self.w0)

def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            print('sine_init for Siren...')
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)

def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            print('first_layer_sine_init for Siren...')
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)

def init_weights(m):
    # if hasattr(modules, 'weight'):
    if isinstance(m, nn.Linear):
        num_input = m.weight.size(-1)
        # See supplement Sec. 1.5 for discussion of factor 30
        m.weight.data.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)

if __name__ == '__main__':
    model = ResMLP(
        in_dim=3,
        hidden_list=[256, 256],
        out_dim=3,
        act='relu',
        block_type='linear_start' # ['relu_start', 'linear_start']
    ).cuda().eval()
    

    x = torch.randn((16, 2304, 3)).cuda()
    x = model(x)
    print(x.shape)
    print(model)



#TODO 正弦位置编码
class PE(nn.Module):
    """
    perform positional encoding
    """
    def __init__(self, P):
        """
        P是位置编码中的超参矩阵
        P: (2, F) encoding matrix
        """
        super().__init__()
        self.register_buffer("P", P)

    @property
    def out_dim(self):
        return self.P.shape[1]*2

    def forward(self, x):
        """
        x: (B, 2)
        """
        x_ = 2*np.pi*x @ self.P # (B, F)
        return torch.cat([torch.sin(x_), torch.cos(x_)], 1) # (B, 2*F)
    

#TODO SIREN（将正弦激活函数显式地加入线性层得到SineLayer）
class SineLayer(nn.Module):
    '''
        See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for
        discussion of omega_0.
    
        If is_first=True, omega_0 is a frequency factor which simply multiplies
        the activations before the nonlinearity. Different signals may require
        different omega_0 in the first layer - this is a hyperparameter.
    
        If is_first=False, then the weights will be divided by omega_0 so as to
        keep the magnitude of activations constant, but boost gradients to the
        weight matrix (see supplement Sec. 1.5)
    '''
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30, scale=10.0, init_weights=True):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        if init_weights:
            self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

@register('sirenet')   
class SirenNet(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features, 
                 hidden_layers, 
                 out_features,
                 outermost_linear=True,
                 first_omega_0=30,
                 hidden_omega_0=30.,
                 scale=10.0,
                 pos_encode=False,
                 sidelength=512,
                 fn_samples=None,
                 use_nyquist=True):
        super().__init__()
        self.pos_encode = pos_encode
        self.nonlin = SineLayer
            
        self.net = []
        self.net.append(self.nonlin(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0,
                                  scale=scale))

        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0,
                                      scale=scale))

        if outermost_linear:
            dtype = torch.float
            final_linear = nn.Linear(hidden_features,
                                     out_features,
                                     dtype=dtype)
            
            with torch.no_grad():
                const = np.sqrt(6/hidden_features)/max(hidden_omega_0, 1e-12)
                final_linear.weight.uniform_(-const, const)
                    
            self.net.append(final_linear)
        else:
            self.net.append(self.nonlin(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0,
                                      scale=scale))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        if self.pos_encode:
            coords = self.positional_encoding(coords)
            
        output = self.net(coords)
                    
        return output

#TODO Gauss
#from https://github.com/vishwa91/wire/blob/main/modules/gauss.py
class GaussLayer(nn.Module):
    '''
        Drop in replacement for SineLayer but with Gaussian non linearity
    '''
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30, scale=10.0):
        '''
            is_first, and omega_0 are not used.
        '''
        super().__init__()
        self.in_features = in_features
        self.omega_0 = omega_0
        self.scale = scale
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, input):
        return torch.exp(-(self.scale*self.linear(input))**2)
    

class GaussNet(nn.Module):
    def __init__(self, in_features,
                 hidden_features, hidden_layers, 
                 out_features,outermost_linear=True,
                 first_omega_0=30, hidden_omega_0=30., scale=10.0,
                 pos_encode=False, sidelength=512, fn_samples=None,
                 use_nyquist=True):
        super().__init__()
        self.pos_encode = pos_encode
        
        self.complex = False
        self.nonlin = GaussLayer
            
        self.net = []
        self.net.append(self.nonlin(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0,
                                  scale=scale))

        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0,
                                      scale=scale))

        if outermost_linear:
            if self.complex:
                dtype = torch.cfloat
            else:
                dtype = torch.float
            final_linear = nn.Linear(hidden_features,
                                     out_features,
                                     dtype=dtype)
                        
            self.net.append(final_linear)
        else:
            self.net.append(self.nonlin(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0,
                                      scale=scale))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        output = self.net(coords)
                    
        return output 


#TODO WIRE
#from https://github.com/vishwa91/wire/blob/main/modules/wire.py
class RealGaborLayer(nn.Module):
    '''
        Implicit representation with Gabor nonlinearity
        
        Inputs;
            in_features: Input features
            out_features; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega: Frequency of Gabor sinusoid term
            scale: Scaling of Gabor Gaussian term
    '''
    
    def __init__(self,
                 in_features, out_features, bias=True,
                 is_first=False, omega0=10.0, sigma0=10.0,
                 trainable=False):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first
        
        self.in_features = in_features
        
        self.freqs = nn.Linear(in_features, out_features, bias=bias)
        self.scale = nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, input):
        omega = self.omega_0 * self.freqs(input)
        scale = self.scale(input) * self.scale_0
        
        return torch.cos(omega)*torch.exp(-(scale**2))

class ComplexGaborLayer(nn.Module):
    '''
        Implicit representation with complex Gabor nonlinearity
        
        Inputs;
            in_features: Input features
            out_features; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega0: Frequency of Gabor sinusoid term
            sigma0: Scaling of Gabor Gaussian term
            trainable: If True, omega and sigma are trainable parameters
    '''
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega0=10.0, sigma0=40.0,
                 trainable=False):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first
        
        self.in_features = in_features
        
        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat
            
        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = nn.Parameter(self.omega_0*torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0*torch.ones(1), trainable)
        
        self.linear = nn.Linear(in_features,
                                out_features,
                                bias=bias,
                                dtype=dtype)
    
    def forward(self, input):
        lin = self.linear(input)
        omega = self.omega_0 * lin
        scale = self.scale_0 * lin
        
        return torch.exp(1j*omega.cpu() - scale.cpu().abs().square()).cuda()
    
class ComplexGaborLayer2D(nn.Module):
    '''
        Implicit representation with complex Gabor nonlinearity with 2D activation function
        
        Inputs;
            in_features: Input features
            out_features; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega0: Frequency of Gabor sinusoid term
            sigma0: Scaling of Gabor Gaussian term
            trainable: If True, omega and sigma are trainable parameters
    '''
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega0=10.0, sigma0=10.0,
                 trainable=False):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first
        
        self.in_features = in_features
        
        #TODO 只有第一层处理float32
        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat
            
        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = nn.Parameter(self.omega_0*torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0*torch.ones(1), trainable)
        
        self.linear = nn.Linear(in_features,
                                out_features,
                                bias=bias,
                                dtype=dtype)
        
        # Second Gaussian window
        self.scale_orth = nn.Linear(in_features,
                                    out_features,
                                    bias=bias,
                                    dtype=dtype)
    
    def forward(self, input):
        lin = self.linear(input)
        
        scale_x = lin
        scale_y = self.scale_orth(input)
        
        #freq_term = torch.exp(1j*self.omega_0*lin)
        freq_term = torch.exp(1j * self.omega_0.cpu() * lin.cpu()).cuda()
        #print('##########################################', freq_term.dtype) #torch.complex64
        scale_x_s = scale_x.cpu().abs().square().cuda()
        scale_y_s = scale_y.cpu().abs().square().cuda()
        arg = scale_x_s + scale_y_s
        #print('##########################################', arg.dtype) #torch.float32
              
        gauss_term = torch.exp(-self.scale_0*self.scale_0*arg)      
        return freq_term*gauss_term #torch.complex64

@register('wirenet')    
class WireNet(nn.Module):
    def __init__(self,
                in_features,
                hidden_features, 
                hidden_layers, 
                out_features,
                outermost_linear=True,
                first_omega_0=10,
                hidden_omega_0=10.,
                scale=10.0,
                pos_encode=False,
                sidelength=512,
                fn_samples=None,
                use_nyquist=True):
        super().__init__()
        
        # All results in the paper were with the default complex 'gabor' nonlinearity
        #TODO [RealGaborLayer, ComplexGaborLayer, ComplexGaborLayer2D]
        self.nonlin = ComplexGaborLayer2D
        
        # Since complex numbers are two real numbers, reduce the number of 
        # hidden parameters by 4

        #TODO 复数具有实部与虚部，信息量是实数的两倍
        #hidden_features = int(hidden_features)
        #hidden_features = int(hidden_features/np.sqrt(2))
        hidden_features = int(hidden_features/2) # 256->128
        
        #TODO [float32, cfloat, cfloat]
        #dtype = torch.float32
        dtype = torch.cfloat # ComplexFloat
        

        self.complex = True
        self.wavelet = 'gabor'    
        
        # Legacy parameter
        self.pos_encode = False


        #BUG 由于网络中存在复数类型的数据输入导致模型优化报错（optimizer.step()）  
        #TODO 升级torch/对复数类型的实部与虚部分别进行优化  
        self.net = []
        # 输入层（输入数据为Float32，is_first=True）
        self.net.append(self.nonlin(in_features,
                                    hidden_features, 
                                    omega0=first_omega_0,
                                    sigma0=scale,
                                    is_first=True,
                                    trainable=False))
        # 隐藏层（输入数据为ComplexFloat，is_first=True）
        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features,
                                        hidden_features, 
                                        omega0=hidden_omega_0,
                                        sigma0=scale))
        # 输出层（输入数据为ComplexFloat，默认is_first=False）
        final_linear = nn.Linear(hidden_features,
                                 out_features,
                                 dtype=dtype)             
        self.net.append(final_linear)
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        output = self.net(coords)
        
        if self.wavelet == 'gabor':
            return output.real
        return output


#TODO MFN
# from https://github.com/boschresearch/multiplicative-filter-networks/blob/main/mfn/mfn.py
class MFNBase(nn.Module):
    """
    Multiplicative filter network base class.

    Expects the child class to define the 'filters' attribute, which should be 
    a nn.ModuleList of n_layers+1 filters with output equal to hidden_size.
    """

    def __init__(
        self, hidden_size, out_size, n_layers, weight_scale, bias=True, output_act=False
    ):
        super().__init__()

        self.linear = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size, bias) for _ in range(n_layers)]
        )
        self.output_linear = nn.Linear(hidden_size, out_size)
        self.output_act = output_act

        for lin in self.linear:
            lin.weight.data.uniform_(
                -np.sqrt(weight_scale / hidden_size),
                np.sqrt(weight_scale / hidden_size),
            )

        return

    def forward(self, x):
        out = self.filters[0](x)
        for i in range(1, len(self.filters)):
            out = self.filters[i](x) * self.linear[i - 1](out)
        out = self.output_linear(out)

        #TODO 最后一层激活
        if self.output_act == 'sin':
            out = torch.sin(out)
        elif self.output_act == 'sigmoid':
            out = nn.Sigmoid(out)

        return out


class FourierLayer(nn.Module):
    """
    Sine filter as used in FourierNet.
    """

    def __init__(self, in_features, out_features, weight_scale):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.linear.weight.data *= weight_scale  # gamma
        self.linear.bias.data.uniform_(-np.pi, np.pi)
        return

    def forward(self, x):
        return torch.sin(self.linear(x))


class FourierNet(MFNBase):
    def __init__(
        self,
        in_features=2,
        hidden_features=256,
        out_features=3,
        hidden_layers=3,
        input_scale=256.0,
        weight_scale=1.0,
        bias=True,
        output_act=False,
    ):
        super().__init__(
            hidden_features, out_features, hidden_layers, weight_scale, bias, output_act
        )
        self.filters = nn.ModuleList(
            [
                FourierLayer(in_features, hidden_features, input_scale / np.sqrt(hidden_layers + 1))
                for _ in range(hidden_layers + 1)
            ]
        )

class GaborLayer(nn.Module):
    """
    Gabor-like filter as used in GaborNet.
    """

    def __init__(self, in_features, out_features, weight_scale, alpha=1.0, beta=1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.mu = nn.Parameter(2 * torch.rand(out_features, in_features) - 1)
        self.gamma = nn.Parameter(
            torch.distributions.gamma.Gamma(alpha, beta).sample((out_features,))
        )
        self.linear.weight.data *= weight_scale * torch.sqrt(self.gamma[:, None])
        self.linear.bias.data.uniform_(-np.pi, np.pi)
        return

    def forward(self, x):
        #TODO 另一种表示方法
        #norm = (input ** 2).sum(dim=1).unsqueeze(-1) + (self.mu ** 2).sum(dim=1).unsqueeze(0) - 2 * input @ self.mu.T
        #return torch.exp(- self.gamma.unsqueeze(0) / 2. * norm) * torch.sin(self.linear(input))

        #TODO 传统表示方法
        #D = torch.norm(rearrange(x, 'b d -> b 1 d')-self.mu, dim=-1)**2
        D = (
            (x ** 2).sum(-1)[..., None]
            + (self.mu ** 2).sum(-1)[None, :]
            - 2 * x @ self.mu.T
        )
        return torch.sin(self.linear(x)) * torch.exp(-0.5 * D * self.gamma[None, :])


#TODO 另一种表示方法对应的GaborNet
# class GaborNet(nn.Module):
#     def __init__(self, in_features=2, hidden_features=256,
#                  hidden_layers=4, out_features=1, 
#                  outermost_linear=True, first_omega_0=0,
#                  hidden_omega_0=0, scale=1, pos_encode=False,
#                  sidelength=1, fn_samples=None, use_nyquist=None):
#         super(INR, self).__init__()

#         self.k = hidden_layers+1
#         self.gabon_filters = nn.ModuleList([GaborLayer(in_features, hidden_features, 0, alpha=6.0 / self.k) for _ in range(self.k)])
#         self.linear = nn.ModuleList(
#             [torch.nn.Linear(hidden_features, hidden_features) for _ in range(self.k - 1)] + [torch.nn.Linear(hidden_features, out_features)])
#         for lin in self.linear[:self.k - 1]:
#             lin.weight.data.uniform_(-np.sqrt(1.0 / hidden_features), np.sqrt(1.0 / hidden_features))

#     def forward(self, x):
#         # Recursion - Equation 3
#         zi = self.gabon_filters[0](x[0, ...])  # Eq 3.a
#         for i in range(self.k - 1):
#             zi = self.linear[i](zi) * self.gabon_filters[i + 1](x[0, ...])
#             # Eq 3.b
#         return self.linear[self.k - 1](zi)[None, ...]  # Eq 3.c
    
@register('gabornet') 
class GaborNet(MFNBase):
    def __init__(
        self,
        in_features=2,
        hidden_features=256,
        out_features=3,
        hidden_layers=3,
        input_scale=256.0,
        weight_scale=1.0,
        alpha=6.0,
        beta=1.0,
        bias=True,
        output_act=False, #['sin', 'sigmoid']
    ):
        super().__init__(
            hidden_features, out_features, hidden_layers, weight_scale, bias, output_act
        )
        self.filters = nn.ModuleList(
            [
                GaborLayer(
                    in_features,
                    hidden_features,
                    input_scale / np.sqrt(hidden_layers + 1),
                    alpha / (hidden_layers + 1),
                    beta,
                )
                for _ in range(hidden_layers + 1)
            ]
        )


#TODO BACON(MFN-based)
# from https://github.com/computational-imaging/bacon/blob/main/modules.py
def mfn_weights_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-(12/num_input)**0.5, (12/num_input)**0.5)


class GaborLayer_Bacon(nn.Module):
    def __init__(self, in_features, out_features, weight_scale, alpha,
                 quantization_interval):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

        self.mu = nn.Parameter(2*torch.rand(1, out_features, in_features)-1)
        self.gamma = nn.Parameter(
            torch.distributions.gamma.Gamma(alpha, 1.0).sample((out_features,)))

        # sample discrete frequencies to ensure coverage
        for i in range(in_features):
            init = torch.randint_like(
                self.linear.weight.data[:, i],
                int(2*weight_scale[i]/quantization_interval)+1)
            init = init * quantization_interval - weight_scale[i]
            self.linear.weight.data[:, i] = init*self.gamma**0.5

        self.linear.weight.requires_grad = False
        self.linear.bias.data.uniform_(-np.pi, np.pi)

    def forward(self, x):
        D = torch.norm(rearrange(x, 'b d -> b 1 d')-self.mu, dim=-1)**2
        return torch.sin(self.linear(x)) * \
               torch.exp(-0.5*D*rearrange(self.gamma, 'o -> 1 o'))


class MultiscaleBACON(nn.Module):
    def __init__(self,
                 in_size=2,
                 hidden_size=256,
                 out_size=3,
                 n_layers=4,
                 alpha=6.0,
                 frequency=(128, 128),
                 quantization_interval=2*np.pi,
                 input_scales=[1/8, 1/8, 1/4, 1/4, 1/4],
                 output_layers=[1, 2, 4]):
        super().__init__()

        self.n_layers = n_layers
        self.output_layers = output_layers

        # we need to multiply by this to be able to fit the signal
        input_scales = [[round((np.pi*freq*s)/quantization_interval) * \
                         quantization_interval
                         for freq in frequency] for s in input_scales]

        self.filters = nn.ModuleList([
                        GaborLayer_Bacon(in_size, hidden_size,
                                         input_scales[i]/np.sqrt(n_layers+1),
                                         alpha/(n_layers+1),
                                         quantization_interval)
                        for i in range(n_layers+1)])
        self.linear = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(n_layers)])
        self.linear.apply(mfn_weights_init)

        self.out = \
            nn.Sequential(nn.Linear(hidden_size, out_size), nn.Sigmoid()) 
            
        # make the final layer (after sigmoid) "almost" uniform in [0, 1]
        # TODO: find the math formula...
        nn.init.uniform_(self.out[0].weight,
                         -6/hidden_size**0.5, 6/hidden_size**0.5)

    def forward(self, x):
        outs = []
        out = self.filters[0](x)
        for i in range(1, len(self.filters)):
            out = self.filters[i](x) * self.linear[i-1](out)
            if i in self.output_layers:
                outs += [self.out(out)]

        return outs


#TODO 后续集成ResidualMFN
# from https://github.com/shekshaa/ResidualMFN/blob/main/modules.py


#TODO 后续集成采用MoE来升级MLP的两个工作：
# from https://github.com/Euijune/Implicit-Neural-Representations-with-levels-of-experts/blob/main/model/model.py
# from https://github.com/VITA-Group/Neural-Implicit-Dict/blob/main/models.py




















# ResMLP(
#   (act): ReLU()
#   (layers): Sequential(
#     (0): Linear(in_features=3, out_features=256, bias=True)
#     (1): ResBlockReLUStart(
#       (act1): ReLU()
#       (linear1): Linear(in_features=256, out_features=256, bias=True)
#       (act2): ReLU()
#       (linear2): Linear(in_features=256, out_features=256, bias=True)
#     )
#     (2): ResBlockReLUStart(
#       (act1): ReLU()
#       (linear1): Linear(in_features=256, out_features=3, bias=True)
#       (act2): ReLU()
#       (linear2): Linear(in_features=3, out_features=3, bias=True)
#       (residual_layer): Linear(in_features=256, out_features=3, bias=True)
#     )
#   )
# )

# ResMLP(
#   (act): ReLU()
#   (layers): Sequential(
#     (0): Linear(in_features=3, out_features=256, bias=True)
#     (1): ResBlockReLUStart(
#       (act1): ReLU()
#       (linear1): Linear(in_features=256, out_features=256, bias=True)
#       (act2): ReLU()
#       (linear2): Linear(in_features=256, out_features=256, bias=True)
#     )
#     (2): ResBlockReLUStart(
#       (act1): ReLU()
#       (linear1): Linear(in_features=256, out_features=3, bias=True)
#       (act2): ReLU()
#       (linear2): Linear(in_features=3, out_features=3, bias=True)
#       (residual_layer): Linear(in_features=256, out_features=3, bias=True)
#     )
#   )
# )