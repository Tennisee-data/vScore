"""
vScore Metaclass System

The metaclass is the scoring structure itself. Any domain where
video maps to outcome axes is a ScoredDomain. The intelligence
lives in the mapping from pixels to scores — language is an
optional lookup table bolted on at the end.

Hierarchy:
    Level 0: Valence vector (numerical scores on domain axes)
    Level 1: Dominant activation cluster (which axes are non-zero)
    Level 2: Learned feature groups (visual patterns the encoder detects)
    Level 3: Language labels (optional, multilingual, late-stage)
"""

from __future__ import annotations
from dataclasses import dataclass, field


class ScoredDomain(type):
    """
    Metaclass for any domain where visual input maps to outcome axes.

    Every domain registered through this metaclass shares the same
    mechanism:
        - Zero is neutral / baseline / homeostasis
        - Non-zero is deviation demanding attention
        - Trajectory over time predicts where scores are heading
        - Threshold crossing triggers action

    The principle is domain-agnostic. Fire, survival, hockey, trading —
    the cognitive operation is identical. Only the axes differ.
    """

    registry: dict[str, type] = {}

    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)

        axes = namespace.get("axes")
        if axes is not None:
            if not isinstance(axes, (list, tuple)):
                raise TypeError(f"{name}.axes must be a list or tuple")
            if len(axes) == 0:
                raise ValueError(f"{name}.axes cannot be empty")

            cls._axis_index = {ax: i for i, ax in enumerate(axes)}
            cls._n_axes = len(axes)
            mcs.registry[name] = cls

        return cls

    @classmethod
    def list_domains(mcs) -> list[str]:
        return list(mcs.registry.keys())

    @classmethod
    def get_domain(mcs, name: str) -> type:
        return mcs.registry[name]


class DomainBase(metaclass=ScoredDomain):
    """
    Base class for concrete domains. Subclass this and define `axes`.
    """

    axes = None  # Subclasses must override

    @classmethod
    def n_axes(cls) -> int:
        return cls._n_axes

    @classmethod
    def axis_names(cls) -> list[str]:
        return list(cls.axes)

    @classmethod
    def axis_index(cls, name: str) -> int:
        return cls._axis_index[name]

    @classmethod
    def zero_state(cls) -> list[float]:
        """Homeostasis. The state the system wants to return to."""
        return [0.0] * cls._n_axes

    @classmethod
    def describe(cls) -> str:
        return f"{cls.__name__}: {cls._n_axes} axes {cls.axes}"
