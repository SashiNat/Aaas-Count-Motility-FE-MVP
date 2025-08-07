# DCNv2 package wrapper using mmcv for CenterTrack compatibility
from .dcn_v2 import DCN, DCNv2

# Also make them available at package level
__all__ = ['DCN', 'DCNv2']
