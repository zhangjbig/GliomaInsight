import functools
import logging
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import modules
from .mri_modules.simple_unet import SimpleUNet
logger = logging.getLogger('base')
####################
# initialize
####################


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    logger.info('Initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(
            weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            'initialization method [{:s}] not implemented'.format(init_type))


####################
# define network
####################


# Generator
def define_G(opt):
    
    model_opt = opt['model']

    if model_opt['which_model_G'] == 'mri':
        from .mri_modules import unet
        from .mri_modules import diffusion as diffusion
    else:
        raise NotImplementedError

    denoisor = unet.UNet(
        in_channel=model_opt['unet']['in_channel'],
        out_channel=model_opt['unet']['out_channel'],
        norm_groups=model_opt['unet']['norm_groups'],
        inner_channel=model_opt['unet']['inner_channel'],
        channel_mults=model_opt['unet']['channel_multiplier'],
        attn_res=model_opt['unet']['attn_res'],
        res_blocks=model_opt['unet']['res_blocks'],
        dropout=model_opt['drop_rate'],
        image_size=model_opt['diffusion']['image_size'],
        version=model_opt['unet']['version']
    )

    denoise_fn_opt = opt['noise_model']['unet']

    # basic unet
    denoise_fn = unet.UNet(
        in_channel=denoise_fn_opt['in_channel'],
        out_channel=denoise_fn_opt['out_channel'],
        norm_groups=denoise_fn_opt['norm_groups'],
        inner_channel=denoise_fn_opt['inner_channel'],
        channel_mults=denoise_fn_opt['channel_multiplier'],
        attn_res=denoise_fn_opt['attn_res'],
        res_blocks=denoise_fn_opt['res_blocks'],
        dropout=0.,
        image_size=model_opt['diffusion']['image_size'],
        version=denoise_fn_opt['version'],
        with_noise_level_emb=False,   
    )

    netG = diffusion.GaussianDiffusion(
        denoisor,
        image_size=model_opt['diffusion']['image_size'],
        channels=model_opt['diffusion']['channels'],
        loss_type=model_opt['loss_type'],
        drop_rate=model_opt['drop_rate'],
        conditional=model_opt['diffusion']['conditional'],
        schedule_opt=model_opt['beta_schedule']['train'],
        denoise_fn=denoise_fn
    )
    
    if opt['phase'] == 'train':
        init_weights(netG, init_type='orthogonal')
    if opt['gpu_ids'] and opt['distributed']:
        assert torch.cuda.is_available()
        netG = nn.DataParallel(netG)
    return netG
