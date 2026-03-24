# config.py
import asyncio
import time
from typing import Dict, List, Optional
import numpy as np

class State:
    """Estado global del servidor."""
    def __init__(self):
        self.HOST = '0.0.0.0'  
        self.PORT = 5000
        
        # Configuración del dataset
        self.dataset_name = None
        self.input_size = None
        
        # Configuración del entrenamiento
        self.expected_workers = 0
        self.max_epochs = 10
        self.learning_rate = 0.1
        
        # Estado del entrenamiento
        self.global_weights = None
        self.X_test = None
        self.y_test = None
        
        # Conexiones de workers
        self.worker_writers: Dict[int, asyncio.StreamWriter] = {}
        self.worker_readers: Dict[int, asyncio.StreamReader] = {}
        
        # Datos de entrenamiento
        self.worker_chunks: Dict[int, np.ndarray] = {}
        self.worker_ready: Dict[int, bool] = {}  # Nuevo: estado de preparación de workers
        self.worker_gradients: Dict[int, Dict] = {}
        self.worker_losses: Dict[int, float] = {}
        
        # Estado de preparación de workers
        self.workers_ready: Dict[int, bool] = {}  # Para trackear si cada worker está listo
        
        # Control de entrenamiento
        self.training_active = True
        self.current_epoch = 0
        self.training_start_time = None
        self.epoch_start_time = None
        
        # Lock para operaciones concurrentes
        self.lock = asyncio.Lock()
    
    def check_all_workers_ready_for_training(self) -> bool:
        """Verifica si todos los workers están listos para entrenar."""
        # Si no hay workers configurados, retornar False
        if not self.worker_writers:
            return False
        
        # Verificar que todos los workers estén marcados como ready
        # Si workers_ready está vacío, asumimos que no están listos
        if len(self.workers_ready) == 0:
            return False
            
        # Verificar que todos los workers esperados estén ready
        for worker_id in range(1, self.expected_workers + 1):
            if not self.workers_ready.get(worker_id, False):
                return False
        
        return True
    
    def mark_worker_ready(self, worker_id: int):
        """Marca un worker como listo para entrenar."""
        self.workers_ready[worker_id] = True
        print(f"[Config] Worker {worker_id} marcado como listo")
    

state = State()