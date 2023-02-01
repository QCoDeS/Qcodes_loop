import logging
from collections.abc import Sequence

import hypothesis.strategies as hst
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from numpy.testing import assert_allclose, assert_array_equal
from qcodes.instrument import ChannelList, ChannelTuple, Instrument, InstrumentChannel
from qcodes.parameters import Parameter
from qcodes.tests.instrument_mocks import DummyChannel, DummyChannelInstrument
from qcodes.validators import Numbers

from qcodes_loop.data.location import FormatLocation
from qcodes_loop.loops import Loop


@pytest.fixture(scope='function', name='dci')
def _make_dci():

    dci = DummyChannelInstrument(name='dci')
    try:
        yield dci
    finally:
        dci.close()

@pytest.fixture(scope="function", name="empty_instrument")
def _make_empty_instrument():

    instr = Instrument(name="dci")

    try:
        yield instr
    finally:
        instr.close()

def test_loop_simple(dci):
    loc_fmt = 'data/{date}/#{counter}_{name}_{date}_{time}'
    rcd = {'name': 'loopSimple'}
    loc_provider = FormatLocation(fmt=loc_fmt, record=rcd)
    loop = Loop(dci.channels[0].temperature.sweep(0, 300, 10),
                0.001).each(dci.A.temperature)
    data = loop.run(location=loc_provider)
    assert_array_equal(data.dci_ChanA_temperature_set.ndarray,
                       data.dci_ChanA_temperature.ndarray)


def test_loop_measure_all_channels(dci):
    p1 = Parameter(name='p1', vals=Numbers(-10, 10), get_cmd=None,
                   set_cmd=None)
    loc_fmt = 'data/{date}/#{counter}_{name}_{date}_{time}'
    rcd = {'name': 'allChannels'}
    loc_provider = FormatLocation(fmt=loc_fmt, record=rcd)
    loop = Loop(p1.sweep(-10, 10, 1), 1e-6).\
        each(dci.channels.temperature)
    data = loop.run(location=loc_provider)
    assert data.p1_set.ndarray.shape == (21, )
    assert len(data.arrays) == 7
    for chan in ['A', 'B', 'C', 'D', 'E', 'F']:
        assert getattr(
            data,
            f'dci_Chan{chan}_temperature'
        ).ndarray.shape == (21,)


def test_loop_measure_channels_individually(dci):
    p1 = Parameter(name='p1', vals=Numbers(-10, 10), get_cmd=None,
                   set_cmd=None)
    loc_fmt = 'data/{date}/#{counter}_{name}_{date}_{time}'
    rcd = {'name': 'channelsIndividually'}
    loc_provider = FormatLocation(fmt=loc_fmt, record=rcd)
    loop = Loop(p1.sweep(-10, 10, 1), 1e-6).each(dci.
                                                 channels[0].temperature,
                                                 dci.
                                                 channels[1].temperature,
                                                 dci.
                                                 channels[2].temperature,
                                                 dci.
                                                 channels[3].temperature)
    data = loop.run(location=loc_provider)
    assert data.p1_set.ndarray.shape == (21, )
    for chan in ['A', 'B', 'C', 'D']:
        assert getattr(
            data, f'dci_Chan{chan}_temperature'
        ).ndarray.shape == (21,)


@given(values=hst.lists(hst.floats(0, 300), min_size=4, max_size=4))
@settings(max_examples=10, deadline=None, suppress_health_check=(HealthCheck.function_scoped_fixture,))
def test_loop_measure_channels_by_name(dci, values):
    p1 = Parameter(name='p1', vals=Numbers(-10, 10), get_cmd=None,
                   set_cmd=None)
    for i in range(4):
        dci.channels[i].temperature(values[i])
    loc_fmt = 'data/{date}/#{counter}_{name}_{date}_{time}'
    rcd = {'name': 'channelsByName'}
    loc_provider = FormatLocation(fmt=loc_fmt, record=rcd)
    loop = Loop(p1.sweep(-10, 10, 1), 1e-6).each(
        dci.A.temperature,
        dci.B.temperature,
        dci.C.temperature,
        dci.D.temperature
    )
    data = loop.run(location=loc_provider)
    assert data.p1_set.ndarray.shape == (21, )
    for i, chan in enumerate(['A', 'B', 'C', 'D']):
        assert getattr(
            data, f'dci_Chan{chan}_temperature'
        ).ndarray.shape == (21,)
        assert getattr(
            data, f'dci_Chan{chan}_temperature'
        ).ndarray.max() == values[i]
        assert getattr(
            data, f'dci_Chan{chan}_temperature'
        ).ndarray.min() == values[i]


@given(loop_channels=hst.lists(hst.integers(0, 3), min_size=2, max_size=2,
                               unique=True),
       measure_channel=hst.integers(0, 3))
@settings(max_examples=10, deadline=800,
          suppress_health_check=(HealthCheck.function_scoped_fixture,))
def test_nested_loop_over_channels(dci, loop_channels, measure_channel):
    channel_to_label = {0: 'A', 1: 'B', 2: 'C', 3: "D"}
    loc_fmt = 'data/{date}/#{counter}_{name}_{date}_{time}'
    rcd = {'name': 'nestedLoopOverChannels'}
    loc_provider = FormatLocation(fmt=loc_fmt, record=rcd)
    loop = Loop(dci.channels[loop_channels[0]].temperature.
                sweep(0, 10, 0.5))
    loop = loop.loop(dci.channels[loop_channels[1]].temperature.
                     sweep(50, 51, 0.1))
    loop = loop.each(dci.channels[measure_channel].temperature)
    data = loop.run(location=loc_provider)

    assert getattr(
        data,
        f'dci_Chan{channel_to_label[loop_channels[0]]}_temperature_set'
    ).ndarray.shape == (21,)
    assert getattr(
        data,
        f'dci_Chan{channel_to_label[loop_channels[1]]}_temperature_set'
    ).ndarray.shape == (21, 11,)
    assert getattr(
        data,
        f'dci_Chan{channel_to_label[measure_channel]}_temperature'
    ).ndarray.shape == (21, 11)

    assert_array_equal(getattr(
        data,
        f'dci_Chan{channel_to_label[loop_channels[0]]}_temperature_set'
    ).ndarray, np.arange(0, 10.1, 0.5))

    expected_array = np.repeat(np.arange(50, 51.01, 0.1).reshape(1, 11),
                               21, axis=0)
    array = getattr(
        data,
        f'dci_Chan{channel_to_label[loop_channels[1]]}_temperature_set'
    ).ndarray
    assert_allclose(array, expected_array)


def test_loop_slicing_multiparameter_raises(dci):
    with pytest.raises(NotImplementedError):
        loop = Loop(dci.A.temperature.sweep(0, 10, 1), 0.1)
        loop.each(dci.channels[0:2].dummy_multi_parameter).run()


def test_loop_multiparameter_by_name(dci):
    loc_fmt = 'data/{date}/#{counter}_{name}_{date}_{time}'
    rcd = {'name': 'multiParamByName'}
    loc_provider = FormatLocation(fmt=loc_fmt, record=rcd)
    loop = Loop(dci.A.temperature.sweep(0, 10, 1), 0.1)
    data = loop.each(dci.A.dummy_multi_parameter)\
        .run(location=loc_provider)
    _verify_multiparam_data(data)
    assert 'multi_setpoint_param_this_setpoint_set' in data.arrays.keys()


def test_loop_multiparameter_by_index(dci):
    loc_fmt = 'data/{date}/#{counter}_{name}_{date}_{time}'
    rcd = {'name': 'loopByIndex'}
    loc_provider = FormatLocation(fmt=loc_fmt, record=rcd)
    loop = Loop(dci.channels[0].temperature.sweep(0, 10, 1),
                0.1)
    data = loop.each(dci.A.dummy_multi_parameter)\
        .run(location=loc_provider)
    _verify_multiparam_data(data)


def test_loop_slicing_arrayparameter(dci):
    loc_fmt = 'data/{date}/#{counter}_{name}_{date}_{time}'
    rcd = {'name': 'loopSlicing'}
    loc_provider = FormatLocation(fmt=loc_fmt, record=rcd)
    loop = Loop(dci.A.temperature.sweep(0, 10, 1), 0.1)
    data = loop.each(dci.channels[0:2].dummy_array_parameter)\
        .run(location=loc_provider)
    _verify_array_data(data, channels=('A', 'B'))


def test_loop_arrayparameter_by_name(dci):
    loc_fmt = 'data/{date}/#{counter}_{name}_{date}_{time}'
    rcd = {'name': 'arrayParamByName'}
    loc_provider = FormatLocation(fmt=loc_fmt, record=rcd)
    loop = Loop(dci.A.temperature.sweep(0, 10, 1), 0.1)
    data = loop.each(dci.A.dummy_array_parameter)\
        .run(location=loc_provider)
    _verify_array_data(data)


def test_loop_arrayparameter_by_index(dci):
    loc_fmt = 'data/{date}/#{counter}_{name}_{date}_{time}'
    rcd = {'name': 'arrayParamByIndex'}
    loc_provider = FormatLocation(fmt=loc_fmt, record=rcd)
    loop = Loop(dci.channels[0].temperature.sweep(0, 10, 1),
                0.1)
    data = loop.each(dci.A.dummy_array_parameter)\
        .run(location=loc_provider)
    _verify_array_data(data)


def test_root_instrument(dci):
    assert dci.root_instrument is dci
    for channel in dci.channels:
        assert channel.root_instrument is dci
        for parameter in channel.parameters.values():
            assert parameter.root_instrument is dci


def test_get_attr_on_empty_channellist_works_as_expected(empty_instrument):
    channels = ChannelTuple(empty_instrument, "channels", chan_type=DummyChannel)
    empty_instrument.add_submodule("channels", channels)

    with pytest.raises(
        AttributeError, match="'ChannelTuple' object has no attribute 'temperature'"
    ):
        _ = empty_instrument.channels.temperature


def test_channel_tuple_call_method_basic_test(dci):
    result = dci.channels.turn_on()
    assert result is None


def test_channel_tuple_call_method_called_as_expected(dci, mocker):

    for channel in dci.channels:
        channel.turn_on = mocker.MagicMock(return_value=1)

    result = dci.channels.turn_on("bar")
    # We never return the result (same for Function)
    assert result is None
    for channel in dci.channels:
        channel.turn_on.assert_called_with("bar")


def _verify_multiparam_data(data):
    assert 'multi_setpoint_param_this_setpoint_set' in data.arrays.keys()
    assert_array_equal(
        data.arrays['multi_setpoint_param_this_setpoint_set'].ndarray,
        np.repeat(np.arange(5., 10).reshape(1, 5), 11, axis=0)
    )
    assert 'dci_ChanA_multi_setpoint_param_this' in data.arrays.keys()
    assert_array_equal(
        data.arrays['dci_ChanA_multi_setpoint_param_this'].ndarray,
        np.zeros((11, 5))
    )
    assert 'dci_ChanA_multi_setpoint_param_this' in data.arrays.keys()
    assert_array_equal(
        data.arrays['dci_ChanA_multi_setpoint_param_that'].ndarray,
        np.ones((11, 5))
    )
    assert 'dci_ChanA_temperature_set' in data.arrays.keys()
    assert_array_equal(
        data.arrays['dci_ChanA_temperature_set'].ndarray,
        np.arange(0, 10.1, 1)
    )


def _verify_array_data(data, channels=('A',)):
    assert 'array_setpoint_param_this_setpoint_set' in data.arrays.keys()
    assert_array_equal(
        data.arrays['array_setpoint_param_this_setpoint_set'].ndarray,
        np.repeat(np.arange(5., 10).reshape(1, 5), 11, axis=0)
    )
    for channel in channels:
        aname = f'dci_Chan{channel}_dummy_array_parameter'
        assert aname in data.arrays.keys()
        assert_array_equal(data.arrays[aname].ndarray, np.ones((11, 5))+1)
    assert 'dci_ChanA_temperature_set' in data.arrays.keys()
    assert_array_equal(
        data.arrays['dci_ChanA_temperature_set'].ndarray,
        np.arange(0, 10.1, 1)
    )
