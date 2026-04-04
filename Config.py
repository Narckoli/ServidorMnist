# Config.py - Servidor
import asyncio
import time
from typing import Dict, List, Optional
import numpy as np

class State:
    """Estado global del servidor."""
    def __init__(self):
        self.HOST = '0.0.0.0'
        self.PORT = 5000

        # Modelo y dataset
        self.model_type   = None   # 'mlp' | 'cnn'
        self.model        = None   # instancia ModelCNN (sólo modo cnn)
        self.dataset_name = None
        self.input_size   = None

        # Hiperparámetros
        self.expected_workers = 0
        self.max_epochs       = 10
        self.learning_rate    = 0.01

        # Estado del entrenamiento
        self.global_weights = None
        self.X_test = None
        self.y_test = None

        # Conexiones
        self.worker_writers: Dict[int, asyncio.StreamWriter] = {}
        self.worker_readers: Dict[int, asyncio.StreamReader] = {}

        # Datos por worker
        self.worker_chunks:    Dict[int, np.ndarray] = {}
        self.worker_ready:     Dict[int, bool]       = {}
        self.worker_gradients: Dict[int, Dict]       = {}

        # loss escalar de la última época (para promediar en Training)
        self.worker_losses:    Dict[int, float]      = {}

        # historial de losses por worker a lo largo de todas las épocas
        # usado por Metrics.py para la gráfica "Loss Individual por Worker"
        self.worker_loss_history: Dict[int, List[float]] = {}

        # Preparación de workers
        self.workers_ready: Dict[int, bool] = {}

        # Métricas globales
        self.train_losses:    List[float] = []
        self.test_losses:     List[float] = []
        self.test_accuracies: List[float] = []
        self.epoch_times:     List[float] = []

        # Control
        self.training_active      = True
        self.current_epoch        = 0
        self.training_start_time  = None
        self.epoch_start_time     = None

        # Sincronización
        self.all_workers_ready = None          # asyncio.Event, creado en training_loop
        self.epoch_events: Dict[int, asyncio.Event] = {}
        self.lock = asyncio.Lock()

    # ── helpers ────────────────────────────────────────────────────────────────

    def check_all_workers_ready_for_training(self) -> bool:
        if not self.worker_writers:
            return False
        for wid in range(1, self.expected_workers + 1):
            if not self.workers_ready.get(wid, False):
                return False
        return True

    def mark_worker_ready(self, worker_id: int):
        self.workers_ready[worker_id] = True
        # Inicializar historial de losses para este worker
        if worker_id not in self.worker_loss_history:
            self.worker_loss_history[worker_id] = []
        print(f"[Config] Worker {worker_id} marcado como listo")

state = State()