# dcn_v2.py - Compatibility module for CenterTrack
from mmcv.ops import DeformConv2dPack as _DeformConv2dPack
from mmcv.ops import ModulatedDeformConv2dPack as _ModulatedDeformConv2dPack
import torch.nn as nn
import torch

class DCN(nn.Module):
    """Deformable Convolution wrapper compatible with original DCNv2"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, deform_groups=None, 
                 deformable_groups=None, bias=True):
        super(DCN, self).__init__()
        
        # Handle both parameter names: deform_groups (mmcv) and deformable_groups (original DCNv2)
        if deformable_groups is not None:
            deform_groups = deformable_groups
        elif deform_groups is None:
            deform_groups = 1
        
        # Use DeformConv2dPack but without bias (mmcv limitation)
        self.dcn = _DeformConv2dPack(
            in_channels=in_channels,
            out_channels=out_channels, 
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            deform_groups=deform_groups,
            bias=False  # Always False for mmcv
        )
        
        # Add bias manually if needed
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        # DeformConv2dPack handles offset generation internally
        out = self.dcn(x)
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out

class DCNv2(nn.Module):
    """Modulated Deformable Convolution v2 wrapper compatible with original DCNv2"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, deform_groups=None,
                 deformable_groups=None, bias=True):
        super(DCNv2, self).__init__()
        
        # Handle both parameter names
        if deformable_groups is not None:
            deform_groups = deformable_groups
        elif deform_groups is None:
            deform_groups = 1
        
        # Use ModulatedDeformConv2dPack but without bias (mmcv limitation)
        self.dcn = _ModulatedDeformConv2dPack(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size, 
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            deform_groups=deform_groups,
            bias=False  # Always False for mmcv
        )
        
        # Add bias manually if needed
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        # ModulatedDeformConv2dPack handles offset and mask generation internally
        out = self.dcn(x)
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out

# Make DCN the default export that CenterTrack expects
__all__ = ['DCN', 'DCNv2']
