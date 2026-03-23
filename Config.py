# servidor/config.py
import asyncio
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

@dataclass
class WorkerInfo:
    def __init__(self, writer, reader, worker_id, dataset_chunk):
        self.writer = writer
        self.reader = reader
        self.worker_id = worker_id
        self.dataset_chunk = dataset_chunk
        self.ready = False
        self.gradients = None
        self.loss = None
        self.epoch_completed = False
    
    def mark_ready(self):
        self.ready = True
    
    def mark_epoch_done(self, gradients, loss):
        self.gradients = gradients
        self.loss = loss
        self.epoch_completed = True
    
    def reset_for_next_epoch(self):
        """Resetear para la siguiente época"""
        self.gradients = None
        self.loss = None
        self.epoch_completed = False

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
    
    # Eventos de sincronización
    all_workers_ready: asyncio.Event = field(default_factory=asyncio.Event)
    training_started: bool = False
    
    # Métricas
    training_start_time: Optional[float] = None
    train_losses: List[float] = field(default_factory=list)
    test_losses: List[float] = field(default_factory=list)
    test_accuracies: List[float] = field(default_factory=list)
    worker_losses: Dict[int, List[float]] = field(default_factory=dict)
    epoch_times: List[float] = field(default_factory=list)
    
    def check_all_workers_ready(self) -> bool:
        """Verifica si todos los workers han completado la época actual."""
        if len(self.workers) < self.expected_workers:
            return False
        return all(w.epoch_completed for w in self.workers.values())
    
    def check_all_workers_ready_for_training(self) -> bool:
        """Verifica si todos los workers completaron su setup."""
        if len(self.workers) < self.expected_workers:
            return False
        return all(w.ready_for_training for w in self.workers.values())
    
    def get_completed_workers_data(self) -> tuple:
        """Obtiene los datos de todos los workers que completaron."""
        all_grads = []
        worker_losses = []
        for w in self.workers.values():
            if w.epoch_completed:
                all_grads.append(w.current_grads)
                worker_losses.append(w.current_loss)
        return all_grads, worker_losses
    
    def reset_epoch_sync(self):
        """Resetea el estado para una nueva época."""
        self.all_workers_ready.clear()
        for worker in self.workers.values():
            worker.reset_for_next_epoch()
    
    def mark_worker_epoch_done(self, worker_id: int):
        """Marca un worker como completado y verifica si todos están listos."""
        if worker_id in self.workers:
            if self.check_all_workers_ready():
                self.all_workers_ready.set()
                print(f"\n🎯 ¡Todos los workers completaron la época {self.current_epoch + 1}!")

# Instancia global
state = TrainingState()