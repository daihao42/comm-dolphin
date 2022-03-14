#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from collections import deque
import numpy as np

import random

class CommRL(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        


