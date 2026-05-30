#
# horus_docker
# Copyright (c) 2026 Temple Compute
#
# MIT License
#
"""
Localization for horus_docker.

Import ``tr`` (aliased as ``_``) in any module that has user-visible strings::

    from horus_docker.i18n import tr as _

    _("Something happened.")
    _("%(n)s item processed", "%(n)s items processed", n=count)
"""

from pathlib import Path

from horus_runtime.i18n import make_translator

tr = make_translator("horus_docker", Path(__file__).parent / "locale")
