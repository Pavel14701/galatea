# galatea/prisms/state_manager.py

from abc import ABC, abstractmethod
import threading
import sqlite3
import json
import time
import pickle
import numpy as np
import torch
from typing import Optional, Any, Dict

class StateManager(ABC):
    """Абстрактный протокол для хранилища состояний Призм."""

    @abstractmethod
    def save(self, user_id: str, prism_name: str, state: Dict[str, Any]) -> None:
        """Сохраняет состояние Призмы."""
        pass

    @abstractmethod
    def load(self, user_id: str, prism_name: str) -> Optional[Dict[str, Any]]:
        """Загружает состояние Призмы. Возвращает None, если состояния нет."""
        pass

    @abstractmethod
    def delete(self, user_id: str, prism_name: str) -> None:
        """Удаляет состояние Призмы."""
        pass


class InMemoryStateManager(StateManager):
    """Хранилище в памяти (для MVP)."""
    def __init__(self):
        self._storage: Dict[str, Dict[str, Dict[str, Any]]] = {}  # user_id -> prism_name -> state
        self._lock = threading.Lock()

    def save(self, user_id: str, prism_name: str, state: Dict[str, Any]) -> None:
        with self._lock:
            self._storage.setdefault(user_id, {})[prism_name] = state

    def load(self, user_id: str, prism_name: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._storage.get(user_id, {}).get(prism_name)

    def delete(self, user_id: str, prism_name: str) -> None:
        with self._lock:
            if user_id in self._storage:
                self._storage[user_id].pop(prism_name, None)


class SQLiteStateManager(StateManager):
    """Хранилище в SQLite (для Этапа 1)."""
    def __init__(self, db_path: str = "galatea_states.db"):
        self.db_path = db_path
        self._init_db()
        self._lock = threading.Lock()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS prism_states (
                user_id TEXT,
                prism_name TEXT,
                state BLOB,
                updated_at INTEGER,
                PRIMARY KEY (user_id, prism_name)
            )
        ''')
        conn.commit()
        conn.close()

    def _serialize(self, state: Dict[str, Any]) -> bytes:
        """Сериализует словарь состояния в байты (через pickle)."""
        return pickle.dumps(state)

    def _deserialize(self, data: bytes) -> Dict[str, Any]:
        """Десериализует байты обратно в словарь."""
        return pickle.loads(data)

    def save(self, user_id: str, prism_name: str, state: Dict[str, Any]) -> None:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            blob = self._serialize(state)
            c.execute(
                'REPLACE INTO prism_states (user_id, prism_name, state, updated_at) VALUES (?, ?, ?, ?)',
                (user_id, prism_name, blob, int(time.time()))
            )
            conn.commit()
            conn.close()

    def load(self, user_id: str, prism_name: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('SELECT state FROM prism_states WHERE user_id=? AND prism_name=?', (user_id, prism_name))
            row = c.fetchone()
            conn.close()
            if row:
                return self._deserialize(row[0])
            return None

    def delete(self, user_id: str, prism_name: str) -> None:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('DELETE FROM prism_states WHERE user_id=? AND prism_name=?', (user_id, prism_name))
            conn.commit()
            conn.close()