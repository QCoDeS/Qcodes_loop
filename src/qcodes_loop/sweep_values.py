"""
Sweep value classes and helpers for use with the Loop.

These classes are being deprecated from qcodes, so we provide
them here for use with the Loop.

Originally from qcodes.parameters.sweep_values
"""

from __future__ import annotations

import io
import math
from collections.abc import Iterator, Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Any, SupportsAbs, cast

import numpy as np
from qcodes.metadatable import Metadatable
from qcodes.parameters.parameter_base import ParameterBase

if TYPE_CHECKING:
    from typing import Self

NumberType = float | int


def named_repr(obj: Any) -> str:
    """Enhance the standard repr() with the object's name attribute."""
    s = f"<{obj.__module__}.{type(obj).__name__}: {obj.name!s} at {id(obj)}>"
    return s


def is_sequence(obj: Any) -> bool:
    """
    Test if an object is a sequence.

    We do not consider strings or unordered collections like sets to be
    sequences, but we do accept iterators (such as generators).
    """
    return isinstance(obj, (Iterator, Sequence, np.ndarray)) and not isinstance(
        obj, (str, bytes, io.IOBase)
    )


def permissive_range(
    start: NumberType, stop: NumberType, step: SupportsAbs[NumberType]
) -> list[NumberType]:
    """
    Returns a range (as a list of values) with floating point steps.
    Always starts at start and moves toward stop, regardless of the
    sign of step.

    Args:
        start: The starting value of the range.
        stop: The end value of the range.
        step: Spacing between the values.

    """
    signed_step = abs(step) * (1 if stop > start else -1)
    # take off a tiny bit for rounding errors
    step_count = math.ceil((stop - start) / signed_step - 1e-10)
    return [start + i * signed_step for i in range(step_count)]


# This is very much related to the permissive_range but more
# strict on the input, start and endpoints are always included,
# and a sweep is only created if the step matches an integer
# number of points.
# numpy is a dependency anyways.
# Furthermore the sweep allows to take a number of points and generates
# an array with endpoints included, which is more intuitive to use in a sweep.
def make_sweep(
    start: float, stop: float, step: float | None = None, num: int | None = None
) -> list[float]:
    """
    Generate numbers over a specified interval.
    Requires ``start`` and ``stop`` and (``step`` or ``num``).
    The sign of ``step`` is not relevant.

    Args:
        start: The starting value of the sequence.
        stop: The end value of the sequence.
        step:  Spacing between values.
        num: Number of values to generate.

    Returns:
        numpy.ndarray: numbers over a specified interval as a ``numpy.linspace``.

    Examples:
        >>> make_sweep(0, 10, num=5)
        [0.0, 2.5, 5.0, 7.5, 10.0]
        >>> make_sweep(5, 10, step=1)
        [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        >>> make_sweep(15, 10.5, step=1.5)
        >[15.0, 13.5, 12.0, 10.5]

    """
    if step and num:
        raise AttributeError("Don't use `step` and `num` at the same time.")
    if (step is None) and (num is None):
        raise ValueError(
            "If you really want to go from `start` to "
            "`stop` in one step, specify `num=2`."
        )
    if step is not None:
        steps = abs((stop - start) / step)
        tolerance = 1e-10
        steps_lo = int(np.floor(steps + tolerance))
        steps_hi = int(np.ceil(steps - tolerance))

        if steps_lo != steps_hi:
            raise ValueError(
                "Could not find an integer number of points for "
                "the the given `start`, `stop`, and `step` "
                f"values. \nNumber of points is {steps_lo + 1:d} or {steps_hi + 1:d}."
            )
        num_steps = steps_lo + 1
    elif num is not None:
        num_steps = num
    else:
        raise ValueError(
            "If you really want to go from `start` to "
            "`stop` in one step, specify `num=2`."
        )

    output_list = np.linspace(start, stop, num=num_steps).tolist()
    return cast("list[float]", output_list)


class SweepValues(Metadatable):
    """
    Base class for sweeping a parameter.

    Must be subclassed to provide the sweep values
    Intended use is to iterate over in a sweep, so it must support:

    >>> .__iter__ # (and .__next__ if necessary).
    >>> .set # is provided by the base class

    Optionally, it can have a feedback method that allows the sweep to pass
    measurements back to this object for adaptive sampling:

    >>> .feedback(set_values, measured_values)

    Todo:
        - Link to adaptive sweep

    Args:
        parameter: the target of the sweep, an object with
         set, and optionally validate methods

        **kwargs: Passed on to Metadatable parent

    Raises:
        TypeError: when parameter is not settable

    See AdaptiveSweep for an example

    example usage:

    >>> for i, value in enumerate(sv):
            sv.set(value)
            sleep(delay)
            vals = measure()
            sv.feedback((i, ), vals) # optional - sweep should not assume
                                     # .feedback exists

    note though that sweeps should only require set and __iter__ - ie
    "for val in sv", so any class that implements these may be used in sweeps.

    That allows things like adaptive sampling, where you don't know ahead of
    time what the values will be or even how many there are.

    """

    def __init__(self, parameter: ParameterBase, **kwargs: Any):
        super().__init__(**kwargs)
        self.parameter = parameter
        self.name = parameter.name
        self._values: list[Any] = []

        # allow has_set=False to override the existence of a set method,
        # but don't require it to be present (and truthy) otherwise
        if not (
            getattr(parameter, "set", None) and getattr(parameter, "has_set", True)
        ):
            raise TypeError(f"parameter {parameter} is not settable")

        self.set = parameter.set

    def validate(self, values: Sequence[Any]) -> None:
        """
        Check that all values are allowed for this Parameter.

        Args:
            values: values to be validated.

        """
        if hasattr(self.parameter, "validate"):
            for value in values:
                self.parameter.validate(value)

    def __iter__(self) -> Iterator[Any]:
        """
        Must be overridden (along with __next__ if this returns self)
        by a subclass to tell how to iterate over these values
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return named_repr(self)


class SweepFixedValues(SweepValues):
    """
    A fixed collection of parameter values to be iterated over during a sweep.

    Args:
        parameter: the target of the sweep, an object with set and
            optionally validate methods

        keys: one or a sequence of items, each of which can be:
            - a single parameter value
            - a sequence of parameter values
            - a slice object, which MUST include all three args

        start: The starting value of the sequence.
        stop: The end value of the sequence.
        step:  Spacing between values.
        num: Number of values to generate.


    A SweepFixedValues object is normally created using :class:`Sweeper`:

    >>> sw = Sweeper(p)
    >>> sv = sw[1.2:2:0.01]  # slice notation
    >>> sv = sw[1, 1.1, 1.3, 1.6]  # explicit individual values
    >>> sv = sw[1.2:2:0.01, 2:3:0.02]  # sequence of slices
    >>> sv = sw[logrange(1,10,.01)]  # some function that returns a sequence

    You can also use list operations to modify these:

    >>> sv += sw[2:3:.01] # (another SweepFixedValues of the same parameter)
    >>> sv += [4, 5, 6] # (a bare sequence)
    >>> sv.extend(sw[2:3:.01])
    >>> sv.append(3.2)
    >>> sv.reverse()
    >>> sv2 = reversed(sv)
    >>> sv3 = sv + sv2
    >>> sv4 = sv.copy()

    note though that sweeps should only require set and __iter__ - ie
    "for val in sv", so any class that implements these may be used in sweeps.
    That allows things like adaptive sampling, where you don't know ahead of
    time what the values will be or even how many there are.

    """

    def __init__(
        self,
        parameter: ParameterBase,
        keys: Any | None = None,
        start: float | None = None,
        stop: float | None = None,
        step: float | None = None,
        num: int | None = None,
    ):
        super().__init__(parameter)
        self._snapshot: dict[str, Any] = {}
        self._value_snapshot: list[dict[str, Any]] = []

        if keys is None:
            if start is None:
                raise ValueError("If keys is None, start needs to be not None.")
            if stop is None:
                raise ValueError("If keys is None, stop needs to be not None.")
            keys = make_sweep(start=start, stop=stop, step=step, num=num)
            self._values = keys
            self._add_linear_snapshot(self._values)

        elif isinstance(keys, slice):
            self._add_slice(keys)
            self._add_linear_snapshot(self._values)

        elif is_sequence(keys):
            for key in keys:
                if isinstance(key, slice):
                    self._add_slice(key)
                elif is_sequence(key):
                    # not sure if we really need to support this (and I'm not
                    # going to recurse any more!) but we will get nested lists
                    # if for example someone does `p[list1, list2]`
                    self._values.extend(key)
                else:
                    # assume a single value
                    self._values.append(key)
            # we dont want the snapshot to go crazy on big data
            if self._values:
                self._add_sequence_snapshot(self._values)

        else:
            # assume a single value
            self._values.append(keys)
            self._value_snapshot.append({"item": keys})

        self.validate(self._values)

    def _add_linear_snapshot(self, vals: list[Any]) -> None:
        self._value_snapshot.append(
            {"first": vals[0], "last": vals[-1], "num": len(vals), "type": "linear"}
        )

    def _add_sequence_snapshot(self, vals: Sequence[Any]) -> None:
        self._value_snapshot.append(
            {
                "min": min(vals),
                "max": max(vals),
                "first": vals[0],
                "last": vals[-1],
                "num": len(vals),
                "type": "sequence",
            }
        )

    def _add_slice(self, slice_: slice) -> None:
        if slice_.start is None or slice_.stop is None or slice_.step is None:
            raise TypeError(
                "all 3 slice parameters are required, " + f"{slice_} is missing some"
            )
        p_range = permissive_range(slice_.start, slice_.stop, slice_.step)
        self._values.extend(p_range)

    def append(self, value: Any) -> None:
        """
        Append a value.

        Args:
            value: new value to append

        """
        self.validate((value,))
        self._values.append(value)
        self._value_snapshot.append({"item": value})

    def extend(self, new_values: Sequence[Any] | SweepFixedValues) -> None:
        """
        Extend sweep with new_values

        Args:
            new_values: new values to append

        Raises:
            TypeError: if new_values is not Sequence, nor SweepFixedValues

        """
        if isinstance(new_values, SweepFixedValues):
            if new_values.parameter is not self.parameter:
                raise TypeError(
                    "can only extend SweepFixedValues of the same parameters"
                )
            # these values are already validated
            self._values.extend(new_values._values)
            self._value_snapshot.extend(new_values._value_snapshot)
        elif is_sequence(new_values):
            self.validate(new_values)
            self._values.extend(new_values)
            self._add_sequence_snapshot(new_values)
        else:
            raise TypeError(f"cannot extend SweepFixedValues with {new_values}")

    def copy(self) -> Self:
        """
        Copy this SweepFixedValues.

        Returns:
            SweepFixedValues of copied values

        """
        new_sv = self.__class__(self.parameter, [])
        # skip validation by adding values and snapshot separately
        # instead of on init
        new_sv._values = self._values[:]
        new_sv._value_snapshot = deepcopy(self._value_snapshot)
        return new_sv

    def reverse(self) -> None:
        """Reverse SweepFixedValues in place."""
        self._values.reverse()
        self._value_snapshot.reverse()
        for snap in self._value_snapshot:
            if "first" in snap and "last" in snap:
                snap["last"], snap["first"] = snap["first"], snap["last"]

    def snapshot_base(
        self,
        update: bool | None = False,
        params_to_skip_update: Sequence[str] | None = None,
    ) -> dict[Any, Any]:
        """
        Snapshot state of SweepValues.

        Args:
            update: Place holder for API compatibility.
            params_to_skip_update: Place holder for API compatibility.

        Returns:
            dict: base snapshot

        """
        self._snapshot["parameter"] = self.parameter.snapshot(update=update)
        self._snapshot["values"] = self._value_snapshot
        return self._snapshot

    def __iter__(self) -> Iterator[Any]:
        return iter(self._values)

    def __getitem__(self, key: slice) -> Any:
        return self._values[key]

    def __len__(self) -> int:
        return len(self._values)

    def __add__(self, other: Sequence[Any] | SweepFixedValues) -> Self:
        new_sv = self.copy()
        new_sv.extend(other)
        return new_sv

    def __iadd__(self, values: Sequence[Any] | SweepFixedValues) -> Self:
        self.extend(values)
        return self

    def __contains__(self, value: float) -> bool:
        return value in self._values

    def __reversed__(self) -> Self:
        new_sv = self.copy()
        new_sv.reverse()
        return new_sv


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

    def __getitem__(self, keys: Any) -> SweepFixedValues:
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
