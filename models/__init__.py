from .models import register, make
# 特征提取主干
from . import edsr, rdn, rcan, swinir, act, srformer, cat, hat, art, cfat, dat, rgt, ipg
from . import mambair, mambairv2, mambairv2light
from . import swinir_ref
from . import sadn
# 轻量级特征提取主干
from . import omnisr
from . import mobilesr
from . import safmn
from . import seemore
# 上采样模块
from . import mlp
from . import positionencoder
from . import liif
from . import misc
from . import lte, ltep
from . import itsrn
from . import btc
from . import lit, clit
from . import sronet, galerkin
from . import mlp_mixer, kan_mixer, lmi
from . import ciaosr
from . import openet
from . import lmciaosr, lmliif, lmlte, lmmlp        # LMF
from . import kanmlp
from . import liif_a, mlp_pw, basis, expansion 
from . import diinn, diinn_decoder
from . import cusm
from . import diif, dlte, c2fmlp
from . import liwt
#TODO 组合创新上采样模块
from . import sronet_lte
from . import ciaosr_liif, ciaosr_abla, ciaosr_diff
from . import liif_csnln, sronet_csnln
from . import cuf
from . import liif_cycle


#TODO 组合创新特征提取主干模块
from . import mamba_cnn, mamba_cnn_1
#TODO 主干消融实验
from . import mamba_cnn_no_fusion, mamba_cnn_no_catten, mam_mam_1, cnn_cnn_1

#BUG 有关RWKV的模块不用就不要import
#from . import rwkvir
#from . import rwkv_cnn
#from . import rwkv_cnn_1

#TODO 有关多对比MRI的模块调用
from . import rct, art_mc, cfat_mc, rcan_mat, dualmambair
from . import mcassr
from . import ciaosr_mc
from . import liif_mc, lte_mc, misc_mc, itsrn_mc, diinn_mc

#TODO 有关多对比MRI模型的探索实验
from . import ciaosr_mc_try1