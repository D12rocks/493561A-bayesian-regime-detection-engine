"""Data source adapters package."""

from src.data.adapters.amfi import AMFIAdapter
from src.data.adapters.base import BaseDataAdapter
from src.data.adapters.rbi import RBIAdapter
from src.data.adapters.sebi_nsdl import SEBIFlowsAdapter
from src.data.adapters.yahoo import YahooFinanceAdapter

__all__ = [
    "BaseDataAdapter",
    "YahooFinanceAdapter",
    "RBIAdapter",
    "SEBIFlowsAdapter",
    "AMFIAdapter",
]
