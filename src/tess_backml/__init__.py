#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import logging
import os
from importlib import metadata

__version__ = metadata.version(__package__)
PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))


# Configure logging
def get_logger():
    """
    Creates a logger
    """
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    return logger


log = get_logger()

from .bkg_data import BackgroundCube  # noqa
from .corrector import ScatterLightCorrector  # noqa

__all__ = ["BackgroundCube", "ScatterLightCorrector"]
