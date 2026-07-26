"""State Manager Module.

This module provides an abstract interface and concrete implementations
for persisting the state of Galatea Prisms (Surprise, Bond, Threat).

The state of each prism is stored as a dictionary keyed by user_id
and prism_name.
Two backends are provided:
    - InMemoryStateManager: ephemeral storage (for MVP / testing).
    - SQLiteStateManager: persistent storage (for Stage 1+).

All implementations are thread-safe.
"""

import pickle
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from typing import Any


class StateManager(ABC):
    """Abstract interface for prism state storage.

    Prisms (Surprise, Bond, Threat) use this to load and save their
    internal state across sessions and users.

    The state is expected to be a dictionary that can be serialized
    (e.g., via pickle or JSON).
    """

    @abstractmethod
    def save(
        self,
        user_id: str,
        prism_name: str,
        state: dict[str, Any]
    ) -> None:
        """Save the prism state.

        Args:
            user_id: Unique identifier for the user.
            prism_name: Name of the prism (e.g., 'surprise', 'bond', 'threat').
            state: Dictionary containing the full prism state.

        """

    @abstractmethod
    def load(self, user_id: str, prism_name: str) -> dict[str, Any] | None:
        """Load the prism state.

        Args:
            user_id: Unique identifier for the user.
            prism_name: Name of the prism.

        Returns:
            The stored state dictionary, or None if no state exists.

        """

    @abstractmethod
    def delete(self, user_id: str, prism_name: str) -> None:
        """Delete the prism state.

        Args:
            user_id: Unique identifier for the user.
            prism_name: Name of the prism.

        """


class InMemoryStateManager(StateManager):
    """In-memory state storage (for MVP and testing).

    All states are held in a nested dictionary:
        storage[user_id][prism_name] = state_dict

    Thread-safe via a threading.Lock.
    """

    def __init__(self) -> None:
        self._storage: dict[str, dict[str, dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def save(  # noqa: D102
        self,
        user_id: str,
        prism_name: str,
        state: dict[str, Any]
    ) -> None:
        with self._lock:
            self._storage.setdefault(user_id, {})[prism_name] = state

    def load(self, user_id: str, prism_name: str) -> dict[str, Any] | None:  # noqa: D102, E501
        with self._lock:
            return self._storage.get(user_id, {}).get(prism_name)

    def delete(self, user_id: str, prism_name: str) -> None:  # noqa: D102
        with self._lock:
            if user_id in self._storage:
                self._storage[user_id].pop(prism_name, None)


class SQLiteStateManager(StateManager):
    """SQLite-based persistent state storage (for Stage 1+).

    States are stored in a single table:
        prism_states (user_id TEXT, prism_name TEXT, state BLOB,
        updated_at INTEGER)
    with a composite primary key (user_id, prism_name).

    Serialisation is done via pickle; this supports arbitrary Python objects
    (including torch.Tensor, but they are converted to lists by the prisms).

    Thread-safe via a threading.Lock.
    """

    def __init__(self, db_path: str = 'galatea_states.db'):
        """Initialise the SQLite state manager.

        Args:
            db_path: Path to the SQLite database file.

        """
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Create the `prism_states` table if it does not exist."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS prism_states (
                user_id TEXT,
                prism_name TEXT,
                state BLOB,
                updated_at INTEGER,
                PRIMARY KEY (user_id, prism_name)
            )
            """
        )
        conn.commit()
        conn.close()

    def _serialize(self, state: dict[str, Any]) -> bytes:
        """Convert the state dictionary to bytes using pickle."""
        return pickle.dumps(state)

    def _deserialize(self, data: bytes) -> dict[str, Any]:
        """Convert bytes back to a dictionary using pickle."""
        return pickle.loads(data)

    def save(  # noqa: D102
        self,
        user_id: str,
        prism_name: str,
        state: dict[str, Any]
    ) -> None:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            blob = self._serialize(state)
            c.execute(
                """
                REPLACE INTO prism_states
                (user_id, prism_name, state, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, prism_name, blob, int(time.time()))
            )
            conn.commit()
            conn.close()

    def load(self, user_id: str, prism_name: str) -> dict[str, Any] | None:   # noqa: D102, E501
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute(
                """
                SELECT state FROM prism_states
                WHERE user_id = ? AND prism_name = ?
                """,
                (user_id, prism_name)
            )
            row = c.fetchone()
            conn.close()
            if row:
                return self._deserialize(row[0])
            return None

    def delete(self, user_id: str, prism_name: str) -> None:  # noqa: D102
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute(
                """
                DELETE FROM prism_states
                WHERE user_id = ? AND prism_name = ?
                """,
                (user_id, prism_name)
            )
            conn.commit()
            conn.close()
