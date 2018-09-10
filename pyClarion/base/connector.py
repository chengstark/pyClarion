'''
Tools for propagating information within a Clarion agent.

Usage
=====

This module defines four concrete classes: ``NodeConnector``, 
``ChannelConnector``, ``SelectorConnector``, and ``EffectorConnector``. 
``NodeConnector`` instances are charged with determining and relaying the output 
of their client nodes to listeners. ``ChannelConnectors`` are charged with 
feeding inputs to client ``Channel`` instances and relaying their outputs to 
listeners. 

Listeners to ``NodeConnector`` instances are expected to be either 
``ChannelConnector`` or ``SelectorConnector`` instances. ``NodeConnector`` 
instances are expected to only listen to ``ChannelConnector`` instances, and 
``EffectorConnector`` instances are expected to only listen to 
``SelectorConnector`` instances.

Example
-------

This example is adapted builds on an example from the documentation of 
``pyClarion.base.channel``.

>>> from pyClarion.base.knowledge import Microfeature, Chunk, Flow, Plicity
>>> from pyClarion.base.junction import MaxJunction
>>> class MyPacket(ActivationPacket):
...     def default_activation(self, key):
...         return 0.0
... 
>>> class MyMaxJunction(MaxJunction[MyPacket]):
... 
...     def __call__(self, *input_maps : MyPacket) -> MyPacket:
...         output = MyPacket()
...         if input_maps:
...             output.update(super().__call__(*input_maps))
...         return output
... 
>>> class MyTopDownPacket(MyPacket):
...     pass
... 
>>> class FineGrainedTopDownChannel(Channel[MyPacket]):
...     """Represents a top-down link between two individual nodes."""
... 
...     def __init__(self, chunk, microfeature, weight):
...         self.chunk = chunk
...         self.microfeature = microfeature
...         self.weight = weight
...
...     def __call__(self, input_map : MyPacket) -> MyTopDownPacket:
...         output = MyTopDownPacket({
...             self.microfeature : input_map[self.chunk] * self.weight
...         })
...         return output
... 
>>> ch = Chunk("APPLE")
>>> mf1 = Microfeature("color", "red")
>>> mf2 = Microfeature("tasty", True)
>>> fl1 = Flow(id=(ch, mf1), plicity=Plicity.Abplicit)
>>> fl2 = Flow(id=(ch, mf2), plicity=Plicity.Abplicit)
>>> channel1 = FineGrainedTopDownChannel(ch, mf1, 1.0)
>>> channel2 = FineGrainedTopDownChannel(ch, mf2, 1.0)
>>> nodes = {
...     ch : NodeConnector(ch, MyMaxJunction()),
...     mf1 : NodeConnector(mf1, MyMaxJunction()),
...     mf2 : NodeConnector(mf2, MyMaxJunction()),
... }
...
>>> flows = {
...     fl1 : FlowConnector(fl1, channel1, MyMaxJunction()),
...     fl2 : FlowConnector(fl2, channel2, MyMaxJunction())
... } 
>>> # We need to connect everything up.
>>> nodes[mf1].add_link(fl1, flows[fl1].get_pull_method())
>>> nodes[mf2].add_link(fl2, flows[fl2].get_pull_method())
>>> flows[fl1].add_link(ch, nodes[ch].get_pull_method())
>>> flows[fl2].add_link(ch, nodes[ch].get_pull_method())
>>> # Check that buffers are empty prior to test
>>> nodes[mf1].output_buffer == MyPacket()
True
>>> nodes[mf2].output_buffer == MyPacket()
True
>>> # Set initial chunk activation.
>>> nodes[ch].add_link('External Input', lambda x: MyPacket({ch : 1.0}))
>>> for connector in nodes.values():
...     connector()
... 
>>> # Propagate activations
>>> for connector in flows.values():
...     connector()
... 
>>> for connector in nodes.values():
...     connector()
... 
>>> # Check that propagation worked
>>> nodes[mf1].output_buffer == MyPacket({mf1 : 1.0})
True
>>> nodes[mf2].output_buffer == MyPacket({mf2 : 1.0})
True

Abstractions
============

The abstract ``Connector`` class enables client objects to listen and react to 
information flows within a Clarion agent. The abstract ``Propagator`` class 
extends the functionality of the ``Connector`` class to enable propagation of 
client outputs to downstream listeners.

Instantiation
-------------

Since ``Connector`` is an abstract class, it cannot be directly instantiated:

>>> Connector(Node())
Traceback (most recent call last):
    ...
TypeError: Can't instantiate abstract class Connector with abstract methods __call__

The same is true of the ``Propagator`` class.

>>> Propagator(Node(), MyMaxJunction())
Traceback (most recent call last):
    ...
TypeError: Can't instantiate abstract class Propagator with abstract methods get_pull_method, propagate

Users are not expected to directly implement the abstract ``__call__`` and 
``propagate`` methods of these classes. Instead, they should work with the 
concrete classes provided by this module.

'''


import abc
import copy
from typing import (
    TypeVar, Generic, Hashable, Callable, Any, Dict, Set, List, Iterable, Union
)
from pyClarion.base.knowledge import Node, Flow
from pyClarion.base.packet import Packet, ActivationPacket, SelectorPacket
from pyClarion.base.channel import Channel
from pyClarion.base.junction import Junction
from pyClarion.base.selector import Selector
from pyClarion.base.effector import Effector

##################
# TYPE VARIABLES #
##################

Ct = TypeVar('Ct', bound=Union[Node, Flow])
It = TypeVar('It', bound=Packet)
Ot = TypeVar('Ot', bound=Packet)


################
# ABSTRACTIONS #
################

class Connector(Generic[It], abc.ABC):
    '''
    Connects client to relevant constructs as a listener.

    This is an abstract class, it cannot be directly instantiated.

    For details, see module documentation.
    '''

    def __init__(self) -> None:

        self.input_links: Dict[Hashable, Callable[..., It]] = dict()
        self.input_buffer: List[It] = list()

    @abc.abstractmethod
    def __call__(self) -> None:
        pass

    def pull(self) -> None:
        
        self.input_buffer = [
            callback() for callback in self.input_links.values()
        ] 

    def add_link(
        self, identifier: Hashable, callback: Callable[..., It]
    ) -> None:

        self.input_links[identifier] = callback

    def drop_link(self, identifier: Hashable):

        del self.input_links[identifier]


class Propagator(Connector[It], Generic[It, Ot]):
    '''
    Allows client to push activation packets to listeners.

    This is an abstract class, it cannot be directly instantiated.

    For details, see module documentation.
    '''
    
    def __init__(self, junction: Junction) -> None:
        
        super().__init__()
        self.junction = junction
        self.output_buffer : Ot = self.propagate()

    def __call__(self) -> None:
        '''Update ``self.output_buffer``.'''

        self.pull()        
        self.output_buffer = self.propagate()

    @abc.abstractmethod
    def propagate(self) -> Ot:
        '''
        Compute and return current output of ``self.client``.
        
        This is an abstract method.
        '''
        pass

    @abc.abstractmethod
    def get_pull_method(self) -> Callable:
        pass


class ActivationPropagator(Propagator[ActivationPacket, ActivationPacket], Generic[Ct]):

    def __init__(self, client: Ct, junction: Junction) -> None:

        super().__init__(junction)
        self.client = client

    def get_pull_method(self) -> Callable:

        return self.get_output

    def get_output(self, nodes: Iterable[Node] = None) -> ActivationPacket:
        '''
        Return current output of ``self``.

        Returns a subpacket of ``self.output_buffer``. Intended to be used as a 
        pull method for listeners of this propagator.

        :param nodes: Nodes to be included in returned subpacket. If ``None``, 
            all nodes in ``self.output_buffer`` will be included. 
        '''

        if not nodes:
            nodes = self.output_buffer.keys()
        return self.output_buffer.subpacket(nodes)


####################
# CONCRETE CLASSES #
####################


class NodeConnector(ActivationPropagator[Node]):
    '''Embeds a node in a network.

    For details, see module documentation.
    '''

    def pull(self) -> None:
        
        self.input_buffer = [
            callback([self.client]) for callback in self.input_links.values()
        ]

    def propagate(self) -> ActivationPacket:
        '''Compute and return current output of client node.

        Passes activation packets stored in ``self.buffer`` through 
        ``self.junction`` and returns the result.
        '''
        
        return self.junction(*self.input_buffer)


class FlowConnector(ActivationPropagator[Flow]):
    '''Embeds an activation channel in a Clarion agent.

    For details, see module documentation.
    '''

    def __init__(
        self, client: Flow, channel: Channel, junction: Junction
    ) -> None:
    
        self.channel = channel
        super().__init__(client, junction)


    def propagate(self) -> ActivationPacket:
        '''Update listeners with new output of client activation channel.

        Passes activation packets stored in ``self.buffer`` through 
        ``self.junction`` and feeds result to ``self.client``. Returns the 
        output of ``self.clent``.
        '''

        return self.channel(self.junction(*self.input_buffer))


class SelectorConnector(Propagator[ActivationPacket, SelectorPacket]):
    '''
    Embeds an action selector in a Clarion agent.

    For details, see module documentation.
    '''

    def __init__(self, selector: Selector, junction: Junction) -> None:

        super().__init__(junction)
        self.selector = selector

    def propagate(self) -> SelectorPacket:
        '''Update listeners with new output of client activation channel.

        Passes activation packets stored in ``self.buffer`` through 
        ``self.junction`` and feeds result to ``self.client``. Returns the 
        output of ``self.clent``.
        '''

        return self.selector(self.junction(*self.input_buffer))

    def get_pull_method(self) -> Callable:

        return self.get_output

    def get_output(self, nodes: Iterable[Node] = None) -> SelectorPacket:
        '''
        Return current output of ``self``.

        Returns a deepcopy of ``self.output_buffer``. Intended to be used as a 
        pull method for listeners of this propagator. 
        '''

        return copy.deepcopy(self.output_buffer)


class EffectorConnector(Connector[SelectorPacket]):
    '''
    Embeds an action effector in a Clarion agent.

    For details, see module documentation.
    '''

    def __init__(self, effector: Effector) -> None:

        super().__init__()
        self.effector = effector

    def __call__(self) -> None:

        self.pull()
        self.effector(*self.input_buffer)