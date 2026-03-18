# config.py
import asyncio
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

@dataclass
class WorkerInfo:
    """Información de un worker conectado."""
    writer: asyncio.StreamWriter
    reader: asyncio.StreamReader
    worker_id: int
    dataset_chunk: np.ndarray
    current_grads: Optional[Dict[str, Any]] = None
    current_loss: Optional[float] = None
    epoch_completed: asyncio.Event = field(default_factory=asyncio.Event)
    
    def mark_epoch_done(self, grads: Dict, loss: float):
        self.current_grads = grads
        self.current_loss = loss
        self.epoch_completed.set()
    
    def reset_for_next_epoch(self):
        self.current_grads = None
        self.current_loss = None
        self.epoch_completed.clear()

@dataclass
class TrainingState:
    """Estado global del entrenamiento."""
    # Configuración de red
    HOST: str = "0.0.0.0"
    PORT: int = 5000
    
    # Parámetros de entrenamiento
    expected_workers: int = 0
    max_epochs: int = 10
    learning_rate: float = 0.1
    current_epoch: int = 0
    
    # Modelo y datos
    global_weights: Optional[Dict[str, np.ndarray]] = None
    X_test: Optional[np.ndarray] = None
    y_test: Optional[np.ndarray] = None
    
    # Workers y sincronización
    workers: Dict[int, WorkerInfo] = field(default_factory=dict)
    all_workers_ready: asyncio.Event = field(default_factory=asyncio.Event)
    
    # Métricas
    training_start_time: Optional[float] = None
    train_losses: List[float] = field(default_factory=list)
    test_losses: List[float] = field(default_factory=list)
    test_accuracies: List[float] = field(default_factory=list)
    worker_losses: Dict[int, List[float]] = field(default_factory=dict)
    epoch_times: List[float] = field(default_factory=list)
    
    def check_all_workers_ready(self) -> bool:
        if len(self.workers) < self.expected_workers:
            return False
        return all(w.epoch_completed.is_set() for w in self.workers.values())
    
    def get_completed_workers_data(self) -> tuple:
        all_grads = []
        worker_losses = []
        for w in self.workers.values():
            if w.epoch_completed.is_set():
                all_grads.append(w.current_grads)
                worker_losses.append(w.current_loss)
        return all_grads, worker_losses

# Instancia global
state = TrainingState()