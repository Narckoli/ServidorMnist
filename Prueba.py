#!/usr/bin/env python3
"""
Servidor de entrenamiento distribuido SINCRONO por epoca con CHECKPOINTING.
Guarda estado cada epoca para recuperacion ante fallos.
"""

import asyncio
import struct
import pickle
import logging
import argparse
import random
import json
import time
import hashlib
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("Server")


# ==================== RESULTADOS ====================
@dataclass
class TrainingResult:
    training_id: str = ""
    start_time: str = ""
    end_time: str = ""
    num_workers: int = 3
    num_classes: int = 10
    batch_size: int = 64
    chunk_size: int = 10
    max_epochs: int = 10
    learning_rate: float = 0.001
    dataset: str = "cifar10"
    total_epochs_completed: int = 0
    total_updates: int = 0
    final_loss: float = 0.0
    final_accuracy: float = 0.0
    total_seconds: float = 0.0
    avg_epoch_seconds: float = 0.0
    worker_hardware: dict = field(default_factory=dict)
    worker_updates: dict = field(default_factory=dict)
    epoch_history: list = field(default_factory=list)
    
    def to_dict(self):
        return asdict(self)
    
    def save(self, output_dir: Path = None):
        if output_dir is None:
            output_dir = Path.home() / "training_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"training_{self.training_id}.json"
        filepath = output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"Resultados guardados en: {filepath}")
        return filepath
    
    def print_summary(self):
        print("\n" + "=" * 70)
        print("RESUMEN DEL ENTRENAMIENTO")
        print("=" * 70)
        print(f"   ID:              {self.training_id}")
        print(f"   Inicio:          {self.start_time}")
        print(f"   Fin:             {self.end_time}")
        print(f"   Duracion:        {self._format_time(self.total_seconds)}")
        print("-" * 70)
        print(f"   Dataset:         {self.dataset.upper()}")
        print(f"   Workers:         {self.num_workers}")
        print(f"   Epocas:          {self.total_epochs_completed} / {self.max_epochs}")
        print(f"   Batch size:      {self.batch_size}")
        print(f"   Learning rate:   {self.learning_rate}")
        print("-" * 70)
        print(f"   Total updates:   {self.total_updates}")
        print(f"   Loss final:      {self.final_loss:.6f}")
        print(f"   Accuracy final:  {self.final_accuracy:.2f}%")
        print("-" * 70)
        print("   Hardware de workers:")
        for wid, hw in self.worker_hardware.items():
            print(f"      Worker {wid}: {hw.get('cpu', 'Unknown')} ({hw.get('tier', '?')})")
        print("=" * 70)
    
    def _format_time(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs:.1f}s"


# ==================== MODELO ====================
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SEBlock(nn.Module):
    def __init__(self, in_ch, se_ratio=0.25):
        super().__init__()
        se_ch = max(1, int(in_ch * se_ratio))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_ch, se_ch, 1)
        self.fc2 = nn.Conv2d(se_ch, in_ch, 1)
        self.act = Swish()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        se = self.pool(x)
        se = self.act(self.fc1(se))
        se = self.sigmoid(self.fc2(se))
        return x * se

class MBConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, expand_ratio, kernel_size, stride, se_ratio, drop_rate=0.0):
        super().__init__()
        self.use_residual = (stride == 1 and in_ch == out_ch)
        hidden_dim = in_ch * expand_ratio
        layers = []
        if expand_ratio != 1:
            layers += [nn.Conv2d(in_ch, hidden_dim, 1, bias=False), nn.BatchNorm2d(hidden_dim), Swish()]
        layers += [nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size//2, groups=hidden_dim, bias=False),
                   nn.BatchNorm2d(hidden_dim), Swish()]
        if se_ratio is not None:
            layers.append(SEBlock(hidden_dim, se_ratio))
        layers += [nn.Conv2d(hidden_dim, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch)]
        self.block = nn.Sequential(*layers)
        self.drop_rate = drop_rate
    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            if self.drop_rate > 0 and self.training:
                out = nn.functional.dropout(out, p=self.drop_rate, training=True)
            out = out + x
        return out

class EfficientNetLite0(nn.Module):
    CONFIG = [(1,16,3,1,None,1),(6,24,3,2,None,2),(6,40,5,2,None,2),(6,80,3,2,0.25,3),(6,112,5,1,0.25,3),(6,192,5,2,0.25,4),(6,320,3,1,None,1)]
    def __init__(self, num_classes=10, width_mult=1.0, depth_mult=1.0, dropout_rate=0.2):
        super().__init__()
        out_ch = int(__import__('math').ceil(32 * width_mult / 8) * 8)
        self.stem = nn.Sequential(nn.Conv2d(3, out_ch, 3, 2, 1, bias=False), nn.BatchNorm2d(out_ch), Swish())
        blocks, in_ch = [], out_ch
        for expand_ratio, out_ch_cfg, kernel_size, stride, se_ratio, num_repeat in self.CONFIG:
            out_ch = int(__import__('math').ceil(out_ch_cfg * width_mult / 8) * 8)
            num_repeat = int(__import__('math').ceil(num_repeat * depth_mult))
            for i in range(num_repeat):
                blocks.append(MBConvBlock(in_ch, out_ch, expand_ratio, kernel_size, stride if i==0 else 1, se_ratio, dropout_rate))
                in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        head_ch = int(__import__('math').ceil(1280 * width_mult / 8) * 8)
        self.head = nn.Sequential(nn.Conv2d(in_ch, head_ch, 1, bias=False), nn.BatchNorm2d(head_ch), Swish(), nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(head_ch, num_classes))
    def forward(self, x):
        x = self.stem(x); x = self.blocks(x); x = self.head(x); x = x.view(x.size(0), -1); x = self.classifier(x); return x


# ==================== DATASET MANAGER ====================
@dataclass
class DistributedDatasetManager:
    total_samples: int = 50000
    batch_size: int = 64
    num_workers: int = 3
    chunk_size: int = 10
    seed: int = 42
    total_batches: int = field(init=False)
    total_chunks: int = field(init=False)
    chunks_per_worker: int = field(init=False)
    
    def __post_init__(self):
        self.total_batches = (self.total_samples + self.batch_size - 1) // self.batch_size
        self.total_chunks = (self.total_batches + self.chunk_size - 1) // self.chunk_size
        self.chunks_per_worker = self.total_chunks // self.num_workers
    
    def generate_epoch_assignment(self, epoch: int) -> dict:
        rng = random.Random(self.seed + epoch)
        all_chunks = list(range(self.total_chunks))
        rng.shuffle(all_chunks)
        assignment = {}
        base = self.chunks_per_worker
        for w in range(self.num_workers):
            start = w * base
            end = start + base if w < self.num_workers - 1 else len(all_chunks)
            assignment[w] = sorted(all_chunks[start:end])
        return assignment


# ==================== PROTOCOLO ====================
async def send_msg(writer, data):
    payload = pickle.dumps(data)
    length = struct.pack('>I', len(payload))
    writer.write(length + payload)
    await writer.drain()

async def recv_msg(reader):
    length_data = await reader.readexactly(4)
    length = struct.unpack('>I', length_data)[0]
    payload = await reader.readexactly(length)
    return pickle.loads(payload)


# ==================== CHECKPOINT MANAGER ====================
class CheckpointManager:
    """
    Gestiona guardado y carga de checkpoints del entrenamiento.
    Guarda: modelo, optimizador, estado de epocas, resultados parciales.
    """
    def __init__(self, checkpoint_dir: Path, training_id: str):
        self.checkpoint_dir = checkpoint_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.training_id = training_id
        self.checkpoint_path = self.checkpoint_dir / f"checkpoint_{training_id}.pt"
        self.metadata_path = self.checkpoint_dir / f"checkpoint_{training_id}.json"
    
    def save(self, model, optimizer, epoch, total_updates, result, dataset_manager):
        """Guarda checkpoint completo del estado del entrenamiento."""
        try:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'total_updates': total_updates,
                'training_id': self.training_id,
                'timestamp': datetime.now().isoformat(),
            }
            torch.save(checkpoint, self.checkpoint_path)
            
            # Metadatos legibles (sin tensores)
            metadata = {
                'training_id': self.training_id,
                'epoch': epoch,
                'total_updates': total_updates,
                'timestamp': datetime.now().isoformat(),
                'result': result.to_dict() if hasattr(result, 'to_dict') else result,
            }
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Checkpoint guardado: Epoca {epoch} | Updates: {total_updates}")
            return True
        except Exception as e:
            logger.error(f"Error guardando checkpoint: {e}")
            return False
    
    def load(self, model, optimizer):
        """Carga checkpoint si existe. Retorna (epoch, total_updates) o (0, 0)."""
        if not self.checkpoint_path.exists():
            logger.info("No hay checkpoint previo. Iniciando desde cero.")
            return 0, 0
        
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint.get('epoch', 0)
            total_updates = checkpoint.get('total_updates', 0)
            
            logger.info(f"Checkpoint cargado: Epoca {epoch} | Updates: {total_updates}")
            logger.info(f"Archivo: {self.checkpoint_path}")
            return epoch, total_updates
        except Exception as e:
            logger.error(f"Error cargando checkpoint: {e}")
            logger.info("Iniciando desde cero.")
            return 0, 0
    
    def list_checkpoints(self):
        """Lista todos los checkpoints disponibles."""
        checkpoints = []
        for f in self.checkpoint_dir.glob("checkpoint_*.json"):
            try:
                with open(f, 'r') as fh:
                    meta = json.load(fh)
                checkpoints.append({
                    'training_id': meta.get('training_id'),
                    'epoch': meta.get('epoch'),
                    'timestamp': meta.get('timestamp'),
                    'metadata_file': str(f),
                    'checkpoint_file': str(f).replace('.json', '.pt')
                })
            except:
                pass
        return sorted(checkpoints, key=lambda x: x['timestamp'], reverse=True)
    
    def cleanup_old_checkpoints(self, keep_last=3):
        """Mantener solo los N checkpoints mas recientes."""
        checkpoints = self.list_checkpoints()
        for old in checkpoints[keep_last:]:
            try:
                Path(old['metadata_file']).unlink(missing_ok=True)
                Path(old['checkpoint_file']).unlink(missing_ok=True)
                logger.info(f"Checkpoint antiguo eliminado: {old['training_id']}")
            except:
                pass


# ==================== SERVIDOR CON CHECKPOINTING ====================
class SyncEpochServer:
    def __init__(self, host='0.0.0.0', port=5000, num_workers=3, 
                 num_classes=10, lr=0.001, dataset='cifar10',
                 batch_size=64, chunk_size=10, max_epochs=10,
                 save_results=True, results_dir=None,
                 resume_from_checkpoint=False):
        self.host = host
        self.port = port
        self.num_workers = num_workers
        self.lr = lr
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.save_results = save_results
        
        if results_dir:
            self.results_dir = Path(results_dir)
        else:
            project_dir = Path(__file__).parent.resolve()
            self.results_dir = project_dir / "Resultados"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = EfficientNetLite0(num_classes=num_classes)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        
        if dataset == 'cifar10':
            total_samples = 50000
        else:
            total_samples = 10000
        self.dataset_manager = DistributedDatasetManager(
            total_samples=total_samples, batch_size=batch_size,
            num_workers=num_workers, chunk_size=chunk_size, seed=42
        )
        
        self.workers = {}
        self.worker_info = {}
        self.lock = asyncio.Lock()
        self.all_connected = asyncio.Event()
        self._training_started = False
        
        # SINCRONIZACION POR EPOCA
        self.current_epoch = 0
        self.epoch_gradients = {}
        self.epoch_losses = {}
        self.epoch_start_times = {}
        self.total_updates = 0
        
        # RESULTADOS
        self.result = TrainingResult()
        self.result.training_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result.num_workers = num_workers
        self.result.num_classes = num_classes
        self.result.batch_size = batch_size
        self.result.chunk_size = chunk_size
        self.result.max_epochs = max_epochs
        self.result.learning_rate = lr
        self.result.dataset = dataset
        
        self.start_time = None
        
        # CHECKPOINTING
        self.checkpoint_manager = CheckpointManager(self.results_dir, self.result.training_id)
        self.resume_from_checkpoint = resume_from_checkpoint
        
        # Estado de interrupcion (Ctrl+C, error, etc.)
        self._interrupted = False
        self._finalized = False

    def load_checkpoint_if_exists(self):
        """Cargar checkpoint previo si existe y se solicito resumen."""
        if self.resume_from_checkpoint:
            # Buscar checkpoints previos del mismo training_id o el mas reciente
            checkpoints = self.checkpoint_manager.list_checkpoints()
            if checkpoints:
                # Usar el mas reciente
                latest = checkpoints[0]
                logger.info(f"Resumiendo desde checkpoint: {latest['training_id']} | Epoca {latest['epoch']}")
                # Recrear checkpoint manager con el ID antiguo
                self.checkpoint_manager = CheckpointManager(self.results_dir, latest['training_id'])
                self.result.training_id = latest['training_id']
                epoch, updates = self.checkpoint_manager.load(self.model, self.optimizer)
                self.current_epoch = epoch
                self.total_updates = updates
                return epoch, updates
        
        # Si no hay checkpoint o no se solicito resumen
        return 0, 0

    async def handle_worker(self, reader, writer):
        addr = writer.get_extra_info('peername')
        worker_id = None
        
        try:
            # 1. Handshake
            msg = await recv_msg(reader)
            if msg['type'] != 'handshake':
                logger.warning(f"Handshake invalido de {addr}")
                return
            
            hardware_info = msg.get('hardware', 'unknown')
            
            async with self.lock:
                # Si es reconexion, reutilizar ID si el worker anterior murio
                worker_id = self._assign_worker_id(addr, hardware_info)
                self.workers[worker_id] = (reader, writer)
                self.worker_info[worker_id] = {
                    'hardware': hardware_info,
                    'connected_at': datetime.now().isoformat(),
                    'addr': addr,
                    'reconnected': self._is_reconnection(addr)
                }
                self.result.worker_hardware[worker_id] = hardware_info
                logger.info(f"Worker {worker_id} conectado | {hardware_info.get('cpu', 'Unknown')} | Reconexion: {self.worker_info[worker_id]['reconnected']}")
            
            # 2. Enviar configuracion + estado actual
            await send_msg(writer, {
                'type': 'assign_id',
                'worker_id': worker_id,
                'total_workers': self.num_workers,
                'batch_size': self.batch_size,
                'chunk_size': self.dataset_manager.chunk_size,
                'total_samples': self.dataset_manager.total_samples,
                'dataset': self.dataset,
                'max_epochs': self.max_epochs,
                'current_epoch': self.current_epoch,  # ← NUEVO: informar epoca actual
                'resuming': self.current_epoch > 0   # ← NUEVO: indicar si es resumen
            })
            
            # 3. Esperar a todos los workers
            if len(self.workers) == self.num_workers:
                self.start_time = time.time()
                self.result.start_time = datetime.now().isoformat()
                logger.info("=" * 70)
                logger.info("TODOS LOS WORKERS CONECTADOS")
                logger.info(f"   Training ID: {self.result.training_id}")
                logger.info(f"   Epoca actual: {self.current_epoch}/{self.max_epochs}")
                logger.info(f"   Inicio: {self.result.start_time}")
                logger.info("=" * 70)
                self.all_connected.set()
            else:
                remaining = self.num_workers - len(self.workers)
                logger.info(f"Esperando {remaining} worker(s)...")
                await self.all_connected.wait()
            
            # 4. Iniciar entrenamiento (solo worker 0 y solo una vez)
            should_start = False
            async with self.lock:
                if worker_id == 0 and not self._training_started:
                    self._training_started = True
                    should_start = True
            
            if should_start:
                # Si hay checkpoint, continuar desde ahi, sino desde epoca 0
                start_epoch = self.current_epoch
                if start_epoch >= self.max_epochs:
                    logger.info("Entrenamiento ya completado segun checkpoint.")
                    await self.broadcast_training_done()
                else:
                    await self.broadcast_epoch_start(epoch=start_epoch)
            
            # 5. Bucle principal: recibir resultados de epocas
            while True:
                msg = await recv_msg(reader)
                
                if msg['type'] == 'epoch_complete':
                    received_epoch = msg['epoch']
                    worker_avg_loss = msg.get('avg_loss', 0.0)
                    worker_grads = msg.get('gradients', {})
                    
                    logger.info(f"Worker {worker_id} reporto epoca {received_epoch} | Loss: {worker_avg_loss:.4f}")
                    
                    async with self.lock:
                        if received_epoch not in self.epoch_gradients:
                            self.epoch_gradients[received_epoch] = {}
                            self.epoch_losses[received_epoch] = {}
                        
                        self.epoch_gradients[received_epoch][worker_id] = worker_grads
                        self.epoch_losses[received_epoch][worker_id] = worker_avg_loss
                        
                        received_count = len(self.epoch_gradients[received_epoch])
                        total_active = len(self.workers)
                        
                        logger.info(f"Progreso epoca {received_epoch}: {received_count}/{total_active} workers")
                        
                        # ¿Todos los workers activos reportaron?
                        if received_count >= total_active:
                            # ===== PROMEDIO Y APLICACION =====
                            self.apply_averaged_gradients(received_epoch)
                            self.total_updates += 1
                            
                            # Calcular loss promedio de la epoca
                            epoch_avg_loss = sum(self.epoch_losses[received_epoch].values()) / len(self.epoch_losses[received_epoch])
                            
                            # Guardar historico
                            epoch_duration = time.time() - self.epoch_start_times.get(received_epoch, 0)
                            self.result.epoch_history.append({
                                'epoch': received_epoch,
                                'avg_loss': round(epoch_avg_loss, 6),
                                'duration_seconds': round(epoch_duration, 2),
                                'timestamp': datetime.now().isoformat()
                            })
                            
                            logger.info(f"Epoca {received_epoch} completada | Loss: {epoch_avg_loss:.4f} | Update #{self.total_updates}")
                            
                            # ===== CHECKPOINT: guardar despues de cada epoca =====
                            self.current_epoch = received_epoch + 1
                            self.checkpoint_manager.save(
                                self.model, self.optimizer,
                                self.current_epoch, self.total_updates,
                                self.result, self.dataset_manager
                            )
                            self.checkpoint_manager.cleanup_old_checkpoints(keep_last=5)
                            
                            # Limpiar buffers
                            del self.epoch_gradients[received_epoch]
                            del self.epoch_losses[received_epoch]
                            
                            # ¿Siguiente epoca o fin?
                            if received_epoch < self.max_epochs - 1:
                                next_epoch = received_epoch + 1
                                await self.broadcast_epoch_start(next_epoch)
                            else:
                                logger.info("Todas las epocas completadas!")
                                await self.finish_training()
                                await self.broadcast_training_done()
                                break
                
                elif msg['type'] == 'heartbeat':
                    await send_msg(writer, {'type': 'heartbeat_ack'})
                    
                elif msg['type'] == 'done':
                    logger.info(f"Worker {worker_id} confirmo finalizacion")
                    break
                    
        except asyncio.IncompleteReadError:
            logger.warning(f"Worker {worker_id} desconectado abruptamente")
        except Exception as e:
            logger.error(f"Error Worker {worker_id}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if worker_id is not None:
                async with self.lock:
                    if worker_id in self.workers:
                        del self.workers[worker_id]
                    # NO eliminar de worker_info para mantener historico
            try:
                writer.close()
                await asyncio.wait_for(writer.wait_closed(), timeout=2.0)
            except (asyncio.TimeoutError, OSError, ConnectionResetError):
                pass

    def _assign_worker_id(self, addr, hardware_info):
        """Asignar ID. Si es reconexion de un worker previo, reutilizar su ID."""
        # Buscar si hay un worker muerto con el mismo hardware/IP
        for wid, info in self.worker_info.items():
            if wid not in self.workers:  # Worker desconectado
                if info.get('addr') == addr or info.get('hardware') == hardware_info:
                    logger.info(f"Reutilizando ID {wid} para reconexion")
                    return wid
        return len(self.workers)
    
    def _is_reconnection(self, addr):
        """Verificar si esta direccion ya se conecto antes."""
        for info in self.worker_info.values():
            if info.get('addr') == addr:
                return True
        return False

    def apply_averaged_gradients(self, epoch):
        """Promedia gradientes de todos los workers y aplica un solo update."""
        workers_grads = self.epoch_gradients[epoch]
        num_contributors = len(workers_grads)
        
        self.optimizer.zero_grad()
        
        for name, param in self.model.named_parameters():
            grads = []
            for wid, wg in workers_grads.items():
                if name in wg:
                    grads.append(wg[name])
            
            if grads:
                avg_grad = sum(grads) / num_contributors
                param.grad = avg_grad
        
        self.optimizer.step()

    async def broadcast_epoch_start(self, epoch: int):
        self.epoch_start_times[epoch] = time.time()
        
        assignment = self.dataset_manager.generate_epoch_assignment(epoch)
        state_dict = self.model.state_dict()
        
        msg = {
            'type': 'epoch_start',
            'epoch': epoch,
            'state_dict': state_dict,
            'global_step': self.total_updates,
            'chunk_assignment': assignment,
            'max_epochs': self.max_epochs
        }
        
        tasks = []
        dead_workers = []
        for wid, (_, w) in list(self.workers.items()):
            try:
                tasks.append(send_msg(w, msg))
            except Exception as e:
                logger.warning(f"No se pudo enviar a worker {wid}: {e}")
                dead_workers.append(wid)
        
        for wid in dead_workers:
            async with self.lock:
                if wid in self.workers:
                    del self.workers[wid]
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.info(f"Epoca {epoch}/{self.max_epochs} iniciada para {len(tasks)} workers")

    async def broadcast_training_done(self):
        msg = {
            'type': 'training_done',
            'total_epochs': self.max_epochs,
            'total_updates': self.total_updates
        }
        tasks = []
        for _, (_, w) in list(self.workers.items()):
            try:
                tasks.append(send_msg(w, msg))
            except Exception:
                pass
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def finish_training(self):
        """Finalizar entrenamiento y guardar resultados."""
        if self._finalized:
            return self.result
        
        self._finalized = True
        logger.info("FINALIZANDO ENTRENAMIENTO...")
        
        end_time = time.time()
        total_seconds = end_time - self.start_time if self.start_time else 0
        
        final_loss = 0.0
        if self.result.epoch_history:
            final_loss = self.result.epoch_history[-1]['avg_loss']
        
        final_accuracy = await self.evaluate_model()
        
        self.result.end_time = datetime.now().isoformat()
        self.result.total_seconds = round(total_seconds, 2)
        self.result.total_epochs_completed = len(self.result.epoch_history)
        self.result.total_updates = self.total_updates
        self.result.final_loss = round(final_loss, 6)
        self.result.final_accuracy = round(final_accuracy, 2)
        
        if self.result.epoch_history:
            total_epoch_time = sum(e.get('duration_seconds', 0) for e in self.result.epoch_history)
            self.result.avg_epoch_seconds = round(total_epoch_time / len(self.result.epoch_history), 2)
        
        for wid, info in self.worker_info.items():
            self.result.worker_updates[wid] = info.get('updates_received', 0)
        
        self.result.print_summary()
        
        if self.save_results:
            filepath = self.result.save(self.results_dir)
            logger.info(f"Resultados finales guardados en: {filepath}")
        
        # Guardar checkpoint final
        self.checkpoint_manager.save(
            self.model, self.optimizer,
            self.current_epoch, self.total_updates,
            self.result, self.dataset_manager
        )
        
        return self.result

    async def evaluate_model(self) -> float:
        try:
            import torchvision
            import torchvision.transforms as transforms
            
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            ])
            
            test_dataset = torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True, transform=transform
            )
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=64, shuffle=False, num_workers=2
            )
            
            self.model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in test_loader:
                    output = self.model(data)
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
            
            accuracy = 100 * correct / total
            logger.info(f"Accuracy en test set: {accuracy:.2f}% ({correct}/{total})")
            return accuracy
            
        except Exception as e:
            logger.warning(f"No se pudo evaluar accuracy: {e}")
            return 0.0

    async def start(self):
        # Cargar checkpoint si existe antes de iniciar servidor
        if self.resume_from_checkpoint:
            loaded_epoch, loaded_updates = self.load_checkpoint_if_exists()
            self.current_epoch = loaded_epoch
            self.total_updates = loaded_updates
        
        server = await asyncio.start_server(self.handle_worker, self.host, self.port)
        addrs = ', '.join(str(sock.getsockname()) for sock in server.sockets)
        
        logger.info("=" * 70)
        logger.info("SERVIDOR SYNC POR EPOCA - CON CHECKPOINTING")
        logger.info(f"   Training ID: {self.result.training_id}")
        if self.current_epoch > 0:
            logger.info(f"   RESUMIENDO desde epoca {self.current_epoch}")
        logger.info(f"   Modelo: EfficientNetLite-0")
        logger.info(f"   Dataset: {self.dataset.upper()}")
        logger.info(f"   Workers: {self.num_workers}")
        logger.info(f"   Batch size: {self.batch_size}")
        logger.info(f"   Epocas: {self.max_epochs}")
        logger.info(f"   Checkpoints en: {self.checkpoint_manager.checkpoint_dir}")
        logger.info(f"   Resultados en: {self.results_dir}")
        logger.info(f"   Escuchando en: {addrs}")
        logger.info("=" * 70)
        
        # Manejar senales de interrupcion para guardar antes de morir
        try:
            async with server:
                await server.serve_forever()
        except asyncio.CancelledError:
            logger.info("Servidor cancelado. Guardando checkpoint de emergencia...")
            if not self._finalized and self.current_epoch > 0:
                self.checkpoint_manager.save(
                    self.model, self.optimizer,
                    self.current_epoch, self.total_updates,
                    self.result, self.dataset_manager
                )
                if self.save_results:
                    self.result.end_time = datetime.now().isoformat()
                    self.result.save(self.results_dir)
            raise


# ==================== MAIN ====================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Servidor Sync por Epoca con Checkpointing')
    parser.add_argument('--host', default='0.0.0.0', help='Host')
    parser.add_argument('--port', type=int, default=5000, help='Puerto')
    parser.add_argument('--workers', type=int, default=3, help='Workers')
    parser.add_argument('--num-classes', type=int, default=10, help='Clases')
    parser.add_argument('--lr', type=float, default=0.001, help='LR')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'synthetic'])
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--chunk-size', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=10, help='EPOCAS TOTALES')
    parser.add_argument('--no-save', action='store_true', help='No guardar resultados')
    parser.add_argument('--results-dir', type=str, help='Directorio para resultados')
    parser.add_argument('--resume', action='store_true', help='Resumir desde checkpoint si existe')
    
    args = parser.parse_args()
    
    server = SyncEpochServer(
        host=args.host,
        port=args.port,
        num_workers=args.workers,
        num_classes=args.num_classes,
        lr=args.lr,
        dataset=args.dataset,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        max_epochs=args.epochs,
        save_results=not args.no_save,
        results_dir=args.results_dir,
        resume_from_checkpoint=args.resume
    )
    
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("Servidor detenido por usuario (Ctrl+C)")
        # El checkpoint de emergencia se guarda en el except del start()
        if not server._finalized and server.current_epoch > 0:
            logger.info("Guardando checkpoint de emergencia...")
            server.checkpoint_manager.save(
                server.model, server.optimizer,
                server.current_epoch, server.total_updates,
                server.result, server.dataset_manager
            )
            if server.save_results:
                server.result.end_time = datetime.now().isoformat()
                server.result.save(server.results_dir)
            logger.info("Checkpoint de emergencia guardado. Puedes resumir con --resume")