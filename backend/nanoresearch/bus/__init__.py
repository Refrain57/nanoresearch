"""Message bus module for decoupled channel-agent communication."""

from nanoresearch.bus.events import InboundMessage, OutboundMessage
from nanoresearch.bus.queue import MessageBus

__all__ = ["MessageBus", "InboundMessage", "OutboundMessage"]
