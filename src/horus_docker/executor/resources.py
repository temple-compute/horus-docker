#
# horus_docker
# Copyright (c) 2026 Temple Compute
#
# MIT License
#
"""
ContainerScope implementation for reading resource usage from a
docker container.
"""

from dataclasses import dataclass

from horus_runtime.core.resources import ResourceScope


@dataclass(frozen=True)
class ContainerScope(ResourceScope):
    """
    The work runs inside a container, not in the spawned process tree.
    """

    kind: str = "container"
    container_id: str | None = None
    #: Set when the id is not known yet but the runtime writes it to this path
    #: on the target once the container starts (e.g. ``docker --cidfile``).
    cidfile: str | None = None
