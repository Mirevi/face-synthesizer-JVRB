import os.path as osp
import sys

import numpy as np
from .io import _load

def make_abs_path(d):
    return osp.join(osp.dirname(osp.realpath(__file__)), d)

class ParamsPack():
	"""Parameter package"""
	def __init__(self):
		try:
			d = osp.join('', sys.modules[__name__].__package__.replace('.', '\\'), '..', '3dmm_data')

			# param_mean and param_std are used for re-whitening
			meta = _load(osp.join(d, 'param_whitening.pkl'))
			self.param_mean = meta.get('param_mean')
			self.param_std = meta.get('param_std')
		except:
			raise RuntimeError('Missing data')