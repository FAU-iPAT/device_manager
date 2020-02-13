#! /opt/conda/bin/python3
""" Singleton class to handle device pinning for tensorflow calculations """

# Copyright 2020 FAU-iPAT (http://ipat.uni-erlangen.de/)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import collections
from typing import List, Union, Optional, Iterable, Any
from contextlib import AbstractContextManager
import tensorflow as tf


_PhysicalDevice = collections.namedtuple("PhysicalDevice", ["name", "device_type"])
_DeviceList = List[_PhysicalDevice]
_DeviceString = Union[str, int, List[Union[str, int]]]
_Strategy = Union[tf.distribute.Strategy]
_StrategyScope = AbstractContextManager


class DeviceManager:
    """
    Singleton class to handle device usage of tensorflow package

    !!! Warning !!!

    This class was designed to work for the current hardware setup of the package author. It may or
    may not work for your system!
    """

    _device: Optional[str] = None
    _device_list: _DeviceList = []
    _strategy: _Strategy = None
    _scope: _StrategyScope = None  # type: ignore
    _growth: bool = False

    @staticmethod
    def _short_names(devices: _DeviceList) -> List[str]:
        """
        Method to map device names into short names for tensorflow strategies

        :param devices: List of devices
        :return: List of short names
        """
        result = []
        for device_name in devices:
            parts = device_name.name.split(':')
            result.append('/{}:{}'.format(parts[-2].lower(), parts[-1]))
        return result

    @staticmethod
    def _to_index(device_identifier: Union[int, str]) -> int:
        """
        Method to convert int/str input to a index

        :param device_identifier: Str or int input to be converted
        :return: Resulting index according to given parameter
        """
        if isinstance(device_identifier, int):
            return device_identifier
        if isinstance(device_identifier, str):
            return {
                'FIRST': 0,
                'SECOND': 1,
                'THIRD': 2,
                'FOURTH': 3,
            }.get(str(device_identifier).upper(), -1)
        return -1

    @classmethod
    def _select_from_available(cls, devices: _DeviceString, available: _DeviceList) -> _DeviceList:
        """
        Method to select the devices in a list according to a given selection list

        :param devices: List of devices to select
        :param available: List of all available devices to select from
        :return: List of selected devices
        :raises IndexError: One index for selection is out of bounds in the list of available devices
        """
        result: _DeviceList = []
        if isinstance(devices, list):
            device_list = devices
        else:
            device_list = [devices]
        for device_identifier in device_list:
            if device_identifier == 'all':
                for avail in available:
                    result.append(avail)
            else:
                idx = cls._to_index(device_identifier)
                if (idx < 0) or (idx >= len(available)):
                    raise IndexError('Device "{}" not available!'.format(device_identifier))
                result.append(available[idx])
        return result

    @classmethod
    def cpu(cls, devices: _DeviceString = 'all') -> None:
        """
        Method to select the CPU (or multiple) as device for processing

        :param devices: Identifier of CPU devices to use
        """
        cls._device = 'cpu'
        available: _DeviceList = tf.config.list_physical_devices('CPU')
        cls._device_list = cls._select_from_available(devices, available)
        cls._strategy = None

        return cls

    @classmethod
    def gpu(cls, devices: _DeviceString = 'all') -> None:
        """
        Method to select the GPU (or multiple) as device for processing

        :param devices: Identifier of GPU devices to use
        """
        cls._device = 'gpu'
        available: _DeviceList = tf.config.list_physical_devices('GPU')
        cls._device_list = cls._select_from_available(devices, available)
        cls._strategy = None

        return cls

    @classmethod
    def _build_strategy(cls) -> None:
        """
        Method to assemble a tensorflow distribute strategy based on previous settings
        """
        if cls._strategy is None:
            # Limit GPU memory usage
            if cls._device == 'gpu':
                gpu_list = tf.config.list_physical_devices('GPU')
                try:
                    for gpu in gpu_list:
                        tf.config.experimental.set_memory_growth(gpu, cls._growth)
                except RuntimeError:
                    pass
            devices = cls._short_names(cls._device_list)
            # Return the scope
            if len(devices) >= 2:
                cls._strategy = tf.distribute.MirroredStrategy(devices=devices)
            elif len(devices) == 1:
                cls._strategy = tf.distribute.OneDeviceStrategy(device=devices[0])
            else:
                cls._strategy = tf.distribute.OneDeviceStrategy(device='/cpu:0')

    @classmethod
    def scope(cls) -> _StrategyScope:
        """
        Method to return a scope based on selected strategy

        :return: Scope to be used in "with" statements
        """
        cls._build_strategy()
        return cls._strategy.scope()

    @property
    def strategy(self) -> _Strategy:
        """
        Property of the currently selected distribute strategy

        :return: Current distribute strategy
        """
        self.__class__._build_strategy()  # pylint: disable=protected-access
        return self.__class__._strategy  # pylint: disable=protected-access

    @property
    def replica(self) -> int:
        """
        Number of replica in strategy property

        :return: Number of replicas in the current distribute strategy
        """
        return self.strategy.num_replicas_in_sync

    @property
    def num(self) -> int:
        """
        Number of replica in strategy property

        :return: Number of replicas in the current distribute strategy
        """
        return self.strategy.num_replicas_in_sync

    def __iter__(self) -> Iterable:
        """
        Magic method for returning an iterator of the object

        :return: Iterator of the list of devices currently selected
        """
        return iter(self.__class__._device_list)

    def __enter__(self) -> _StrategyScope:
        """
        Magic method for entering as context manager

        :return: Reference to the context manager
        """
        self.__class__._scope = self.__class__.scope()
        return self.__class__._scope.__enter__()

    def __exit__(self, *exc: Any) -> Optional[bool]:
        """
        Magic method for exiting as context manager

        :param exc: Parameters for the context manager callback method
        :return: Whether possible exceptions have been handled
        """
        return self.__class__._scope.__exit__(*exc)

    @property
    def growth(self) -> bool:
        """
        Dynamic memory growth property

        :return: Current value of the dynamic growth property
        """
        return self.__class__._growth  # pylint: disable=protected-access

    @growth.setter
    def growth(self, value: bool) -> None:
        """
        Property setter for dynamic memory growth

        :param value: New value for dynamic growth
        """
        self.__class__._growth = value  # pylint: disable=protected-access

    def use_cpu(self, foo: callable) -> callable:
        """
        device manager decorator cpu

        :param foo: decorated function
        :return: wrapper
        """
        def wrapper(*args, **kwargs):
            self.cpu()
            with self as device:
                return foo(*args, **kwargs)

        return wrapper

    def use_gpu0(self, foo: callable) -> callable:
        """
        device manager decorator gpu 0

        :param foo: decorated function
        :return: wrapper
        """

        def wrapper(*args, **kwargs):
            self.gpu(0)
            with self as device:
                return foo(*args, **kwargs)

        return wrapper

    def use_gpu1(self, foo: callable) -> callable:
        """
        device manager decorator gpu 1

        :param foo: decorated function
        :return: wrapper
        """

        def wrapper(*args, **kwargs):
            self.gpu(1)
            with self as device:
                return foo(*args, **kwargs)

        return wrapper

    def use_gpu2(self, foo: callable) -> callable:
        """
        device manager decorator gpu 2

        :param foo: decorated function
        :return: wrapper
        """

        def wrapper(*args, **kwargs):
            self.gpu(2)
            with self as device:
                return foo(*args, **kwargs)

        return wrapper

    def use_gpu3(self, foo: callable) -> callable:
        """
        device manager decorator gpu 3

        :param foo: decorated function
        :return: wrapper
        """

        def wrapper(*args, **kwargs):
            self.gpu(3)
            with self as device:
                return foo(*args, **kwargs)

        return wrapper


device_manager: DeviceManager = DeviceManager()
