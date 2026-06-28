"""Chat channels module with plugin architecture."""

from nanoresearch.channels.base import BaseChannel
from nanoresearch.channels.manager import ChannelManager

__all__ = ["BaseChannel", "ChannelManager"]
