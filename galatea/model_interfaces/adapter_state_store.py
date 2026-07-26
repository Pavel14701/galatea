"""Adapter State Store.

This module provides a simple in-memory store for adapter parameters (weights).
It is used by the HuggingFaceAdapterManager to keep a copy of each adapter's
parameters for quick retrieval, saving, or restoration.

The store is designed to be extensible: it can be replaced by a persistent
storage backend (e.g., Redis, SQLite, file system) by implementing the same
interface. This allows adapters to survive server restarts and be shared
across multiple replicas.

Key responsibilities:
- Store parameter dictionaries keyed by adapter name.
- Retrieve parameters by name.
- Delete parameters by name.
- List all stored adapter names.
"""

import torch


class AdapterStateStore:
    """In-memory state store for adapter parameters.

    This class holds a dictionary mapping adapter names to their parameter
    dictionaries (parameter name -> torch.Tensor). It provides basic CRUD
    operations and is thread-safe only if used with external locking.

    The primary use case is to provide a fast, ephemeral cache for adapter
    parameters. For long-term persistence, this class can be subclassed or
    replaced with a storage backend that writes to disk or a remote database.

    Attributes:
        _weights: Internal dictionary {adapter_name: {param_name: tensor}}.

    """

    def __init__(self) -> None:
        """Initialise an empty in-memory adapter state store."""
        # Internal storage: key = adapter name,
        # value = dict of parameter tensors.
        self._weights: dict[str, dict[str, torch.Tensor]] = {}

    def save(self, name: str, params: dict[str, torch.Tensor]) -> None:
        """Save (or overwrite) the parameters of an adapter.

        This method stores a reference to the given parameter dictionary.
        The dictionary is expected to map parameter names (as strings) to
        torch.Tensor objects.

        Args:
            name: The name of the adapter (e.g., "user_123").
            params: A dictionary of parameter names to tensor values.
                    The caller must ensure that the tensors are on the same
                    device and have the correct shapes.

        Notes:
            - This method overwrites any existing entry with the same name.
            - The stored dictionary is not copied; if the caller modifies it
                later, the stored values will also change. To avoid unintended
                side effects, it is recommended to pass a copy or to
                clone tensors before storing.

        """
        self._weights[name] = params

    def load(self, name: str) -> dict[str, torch.Tensor] | None:
        """Load the parameters of an adapter by name.

        Args:
            name: The name of the adapter.

        Returns:
            The parameter dictionary if the adapter exists, otherwise None.

        Notes:
            The returned dictionary is a reference to the stored one.
            Callers should treat it as read-only unless they know the store
            will not be modified concurrently.

        """
        return self._weights.get(name)

    def delete(self, name: str) -> None:
        """Delete the parameters of an adapter from the store.

        If the adapter does not exist, this method does nothing
        (no error is raised).

        Args:
            name: The name of the adapter to delete.

        """
        self._weights.pop(name, None)

    def list(self) -> list[str]:
        """Return a list of all adapter names currently stored.

        Returns:
            A list of adapter name strings.

        Notes:
            The order of the names is not guaranteed and may vary
            between calls.

        """
        return list(self._weights.keys())
