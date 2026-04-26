"""
Sweeper class that provides sweep and __getitem__ methods for parameters.

These methods are being deprecated from qcodes Parameter, so we provide
them here for use with the Loop.
"""

from __future__ import annotations

from qcodes.parameters import SweepFixedValues
from qcodes.parameters.parameter_base import ParameterBase


class Sweeper:
    """
    Wraps a parameter to provide ``sweep`` and ``__getitem__`` methods
    for use with :class:`qcodes_loop.loops.Loop`.

    Args:
        parameter: The parameter to sweep over.

    Examples:
        >>> sw = Sweeper(my_param)
        >>> loop = Loop(sw.sweep(0, 10, step=1), delay=0.1)
        >>> loop = Loop(sw[0:10:1], delay=0.1)
        >>> loop = Loop(sw[0, 0.5, 1.0, 1.5], delay=0.1)
    """

    def __init__(self, parameter: ParameterBase) -> None:
        self.parameter = parameter

    def sweep(
        self,
        start: float,
        stop: float,
        step: float | None = None,
        num: int | None = None,
    ) -> SweepFixedValues:
        """
        Create a collection of parameter values to be iterated over.
        Requires ``start`` and ``stop`` and (``step`` or ``num``).
        The sign of ``step`` is not relevant.

        Args:
            start: The starting value of the sequence.
            stop: The end value of the sequence.
            step: Spacing between values.
            num: Number of values to generate.

        Returns:
            SweepFixedValues: Collection of parameter values to be
            iterated over.

        Examples:
            >>> sw = Sweeper(p)
            >>> sw.sweep(0, 10, num=5)
             [0.0, 2.5, 5.0, 7.5, 10.0]
            >>> sw.sweep(5, 10, step=1)
            [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            >>> sw.sweep(15, 10.5, step=1.5)
            [15.0, 13.5, 12.0, 10.5]
        """
        return SweepFixedValues(
            self.parameter, start=start, stop=stop, step=step, num=num
        )

    def __getitem__(self, keys):
        """
        Slice a Parameter to get a SweepFixedValues object
        to iterate over during a sweep.

        Args:
            keys: one or a sequence of items, each of which can be:
                - a single parameter value
                - a sequence of parameter values
                - a slice object (which must include start, stop, and step)

        Returns:
            SweepFixedValues

        Examples:
            >>> sw = Sweeper(p)
            >>> sv = sw[1.2:2:0.01]       # slice notation
            >>> sv = sw[1, 1.1, 1.3, 1.6] # explicit individual values
            >>> sv = sw[1.2:2:0.01, 2:3:0.02]  # sequence of slices
        """
        return SweepFixedValues(self.parameter, keys)
