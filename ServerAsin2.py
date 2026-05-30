"""
Servidor 100% ASINCRONO.
- NO espera a ningun worker
- Aplica gradientes tan pronto llegan (con staleness weighting)
- Workers trabajan a su ritmo, nunca se bloquean
- Version tracking del modelo para manejar gradientes obsoletos
"""

import asyncio
import struct
import pickle
import logging
import argparse
import random
import json
import time
import signal
import sys
import queue
import threading
from collections import deque
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
logger = logging.getLogger("ServerAsync")


# ==================== RESULTADOS ====================
@dataclass
class TrainingResult:
    training_id: str = ""
    start_time: str = ""
    end_time: str = ""
    num_workers: int = 2
    num_classes: int = 10
    batch_size: int = 16
    max_updates: int = 1000
    learning_rate: float = 0.001
    dataset: str = "cifar10"
    total_updates: int = 0
    final_loss: float = 0.0
    final_accuracy: float = 0.0
    total_seconds: float = 0.0
    avg_update_seconds: float = 0.0
    worker_hardware: dict = field(default_factory=dict)
    worker_updates: dict = field(default_factory=dict)
    worker_staleness_avg: dict = field(default_factory=dict)
    update_history: list = field(default_factory=list)
    staleness_distribution: list = field(default_factory=list)
    
    def to_dict(self):
        return asdict(self)
    
    def save(self, output_dir: Path = None):
        if output_dir is None:
            output_dir = Path.home() / "training_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"training_async_{self.training_id}.json"
        filepath = output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"Resultados guardados en: {filepath}")
        return filepath
    
    def print_summary(self):
        print("\\n" + "=" * 70)
        print("RESUMEN DEL ENTRENAMIENTO ASINCRONO")
        print("=" * 70)
        print(f"   ID:              {self.training_id}")
        print(f"   Inicio:          {self.start_time}")
        print(f"   Fin:             {self.end_time}")
        print(f"   Duracion:        {self._format_time(self.total_seconds)}")
        print("-" * 70)
        print(f"   Dataset:         {self.dataset.upper()}")
        print(f"   Workers:         {self.num_workers}")
        print(f"   Updates:         {self.total_updates} / {self.max_updates}")
        print(f"   Batch size:      {self.batch_size}")
        print(f"   Learning rate:   {self.learning_rate}")
        print("-" * 70)
        print(f"   Loss final:      {self.final_loss:.6f}")
        print(f"   Accuracy final:  {self.final_accuracy:.2f}%")
        print(f"   Tiempo/update:   {self.avg_update_seconds:.3f}s")
        print("-" * 70)
        print("   Hardware de workers:")
        for wid, hw in self.worker_hardware.items():
            updates = self.worker_updates.get(wid, 0)
            staleness = self.worker_staleness_avg.get(wid, 0.0)
            print(f"      Worker {wid}: {hw.get('cpu', 'Unknown')} | "
                  f"Updates: {updates} | Staleness avg: {staleness:.2f}")
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


# ==================== SERVIDOR 100% ASINCRONO ====================
class AsyncServer:
    """
    Servidor completamente asincrono:
    - Recibe gradientes de CUALQUIER worker en CUALQUIER momento
    - Aplica inmediatamente con staleness weighting
    - NUNCA espera a ningun worker
    - Workers reciben modelo actualizado instantaneamente
    """
    
    def __init__(self, host='0.0.0.0', port=5000, num_workers=2, 
                 num_classes=10, lr=0.001, dataset='cifar10',
                 batch_size=16, max_updates=1000,
                 save_results=True, results_dir=None,
                 max_staleness=10,  # Rechazar gradientes con staleness > max_staleness
                 staleness_penalty='linear'):  # 'linear', 'exponential', 'constant'
        
        self.host = host
        self.port = port
        self.num_workers = num_workers
        self.lr = lr
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_updates = max_updates
        self.max_staleness = max_staleness
        self.staleness_penalty = staleness_penalty
        self.save_results = save_results
        
        if results_dir:
            self.results_dir = Path(results_dir)
        else:
            project_dir = Path(__file__).parent.resolve()
            self.results_dir = project_dir / "Resultados"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Modelo global
        self.model = EfficientNetLite0(num_classes=num_classes)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        
        # Version tracking - CADA parametro tiene un contador de version
        self.model_version = 0
        self.param_versions = {}
        for name, param in self.model.named_parameters():
            self.param_versions[name] = 0
        
        # Workers
        self.workers = {}  # worker_id -> (reader, writer)
        self.worker_info = {}
        self.lock = asyncio.Lock()
        self.all_connected = asyncio.Event()
        self._training_started = False
        self._training_done = asyncio.Event()
        
        # Estadisticas por worker
        self.worker_update_count = {}
        self.worker_staleness_sum = {}
        self.worker_last_version = {}  # Ultima version que vio cada worker
        
        # Cola de gradientes (thread-safe para el aplicador)
        self.gradient_queue = asyncio.Queue(maxsize=1000)
        self.total_updates = 0
        self.update_times = deque(maxlen=100)
        
        # Resultados
        self.result = TrainingResult()
        self.result.training_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result.num_workers = num_workers
        self.result.num_classes = num_classes
        self.result.batch_size = batch_size
        self.result.max_updates = max_updates
        self.result.learning_rate = lr
        self.result.dataset = dataset
        
        self.start_time = None
        self.last_eval_time = 0
        self.eval_interval = 50  # Evaluar cada 50 updates
        
        # Checkpoint
        self.checkpoint_dir = self.results_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.checkpoint_dir / f"checkpoint_async_{self.result.training_id}.pt"
        self._saved_emergency = False

    def save_emergency_checkpoint(self):
        if self._saved_emergency:
            return True
        try:
            logger.warning("=" * 60)
            logger.warning("GUARDANDO CHECKPOINT DE EMERGENCIA...")
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'model_version': self.model_version,
                'param_versions': self.param_versions,
                'total_updates': self.total_updates,
                'training_id': self.result.training_id,
                'update_history': self.result.update_history,
                'timestamp': datetime.now().isoformat(),
            }
            torch.save(checkpoint, self.checkpoint_path)
            self._saved_emergency = True
            logger.warning(f"Checkpoint guardado: {self.checkpoint_path}")
            logger.warning("=" * 60)
            return True
        except Exception as e:
            logger.error(f"ERROR guardando checkpoint: {e}")
            return False

    def setup_signal_handlers(self):
        def signal_handler(signum, frame):
            sig_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
            logger.warning(f"SENAL RECIBIDA: {sig_name}")
            self.save_emergency_checkpoint()
            raise KeyboardInterrupt(f"Signal {sig_name}")
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def compute_staleness_weight(self, staleness):
        """
        Calcula el peso de un gradiente segun su staleness.
        - staleness=0 (gradiente fresco): peso = 1.0
        - staleness alto: peso reducido
        """
        if staleness <= 0:
            return 1.0
        
        if self.staleness_penalty == 'linear':
            # Pesos: 1.0, 0.9, 0.8, 0.7, ... hasta 0.1 minimo
            weight = max(0.1, 1.0 - (staleness * 0.1))
        elif self.staleness_penalty == 'exponential':
            # Pesos: 1.0, 0.5, 0.25, 0.125, ...
            weight = max(0.05, 0.5 ** staleness)
        elif self.staleness_penalty == 'constant':
            # Sin penalizacion (peligroso pero util para comparar)
            weight = 1.0
        else:
            weight = max(0.1, 1.0 - (staleness * 0.1))
        
        return weight

    def apply_gradient_update(self, gradients, worker_id, worker_version, loss):
        """
        Aplica UN gradiente al modelo global con staleness weighting.
        Se llama inmediatamente al recibir un gradiente.
        """
        # Calcular staleness: diferencia entre version actual y version usada por el worker
        staleness = self.model_version - worker_version
        
        # Rechazar si es demasiado viejo
        if staleness > self.max_staleness:
            logger.warning(f"Worker {worker_id}: gradiente RECHAZADO (staleness={staleness} > max={self.max_staleness})")
            return False, staleness
        
        # Calcular peso segun staleness
        weight = self.compute_staleness_weight(staleness)
        
        # Aplicar gradiente
        self.optimizer.zero_grad()
        
        valid_params = 0
        for name, param in self.model.named_parameters():
            if name in gradients:
                # Aplicar peso por staleness
                weighted_grad = gradients[name] * weight
                param.grad = weighted_grad
                valid_params += 1
                # Actualizar version de este parametro
                self.param_versions[name] = self.model_version + 1
        
        if valid_params > 0:
            self.optimizer.step()
            self.model_version += 1
            self.total_updates += 1
            
            # Estadisticas
            self.worker_update_count[worker_id] = self.worker_update_count.get(worker_id, 0) + 1
            self.worker_staleness_sum[worker_id] = self.worker_staleness_sum.get(worker_id, 0) + staleness
            
            # Logging cada 10 updates
            if self.total_updates % 10 == 0:
                logger.info(f"Update #{self.total_updates} | Worker {worker_id} | "
                           f"Staleness: {staleness} | Weight: {weight:.3f} | Loss: {loss:.4f}")
            
            return True, staleness
        
        return False, staleness

    async def gradient_processor_task(self):
        """
        Tarea asincrona que procesa gradientes de la cola continuamente.
        Corre en paralelo con las conexiones de workers.
        """
        logger.info("Iniciando procesador de gradientes...")
        
        while self.total_updates < self.max_updates and not self._training_done.is_set():
            try:
                # Esperar gradiente con timeout para poder checkear condiciones
                grad_data = await asyncio.wait_for(self.gradient_queue.get(), timeout=1.0)
                
                update_start = time.time()
                success, staleness = self.apply_gradient_update(
                    grad_data['gradients'],
                    grad_data['worker_id'],
                    grad_data['model_version'],
                    grad_data['loss']
                )
                update_time = time.time() - update_start
                self.update_times.append(update_time)
                
                if success:
                    # Guardar en historial cada 5 updates
                    if self.total_updates % 5 == 0:
                        self.result.update_history.append({
                            'update': self.total_updates,
                            'worker_id': grad_data['worker_id'],
                            'loss': round('loss', 6) if 'loss' in dir() else 0.0,
                            'staleness': staleness,
                            'model_version': self.model_version,
                            'timestamp': datetime.now().isoformat()
                        })
                    
                    # Evaluar modelo periodicamente
                    if self.total_updates % self.eval_interval == 0 and self.total_updates > 0:
                        await self.evaluate_and_log()
                
                # Notificar al worker que su gradiente fue procesado (enviar modelo actualizado)
                await self.send_model_to_worker(grad_data['worker_id'])
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error en gradient processor: {e}")
        
        logger.info(f"Gradient processor terminado. Total updates: {self.total_updates}")

    async def send_model_to_worker(self, worker_id):
        """Envia el modelo actualizado a un worker especifico."""
        if worker_id not in self.workers:
            return
        
        try:
            _, writer = self.workers[worker_id]
            
            # Solo enviar pesos entrenables (no BatchNorm stats)
            trainable_state = {}
            for name, param in self.model.named_parameters():
                trainable_state[name] = param.detach().cpu().clone()
            
            msg = {
                'type': 'model_update',
                'model_version': self.model_version,
                'state_dict': trainable_state,
                'total_updates': self.total_updates,
                'max_updates': self.max_updates
            }
            
            await send_msg(writer, msg)
            self.worker_last_version[worker_id] = self.model_version
            
        except Exception as e:
            logger.warning(f"Error enviando modelo a worker {worker_id}: {e}")

    async def broadcast_model_to_all(self):
        """Envia modelo actualizado a TODOS los workers conectados."""
        tasks = []
        for worker_id in list(self.workers.keys()):
            tasks.append(self.send_model_to_worker(worker_id))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def handle_worker(self, reader, writer):
        """
        Maneja la conexion con un worker.
        - Handshake inicial
        - Recibe gradientes continuamente
        - NUNCA bloquea al worker
        """
        addr = writer.get_extra_info('peername')
        worker_id = None
        
        try:
            # Handshake
            msg = await recv_msg(reader)
            if msg['type'] != 'handshake':
                logger.warning(f"Handshake invalido de {addr}")
                return
            
            hardware_info = msg.get('hardware', 'unknown')
            
            async with self.lock:
                worker_id = len(self.workers)
                self.workers[worker_id] = (reader, writer)
                self.worker_info[worker_id] = {
                    'hardware': hardware_info,
                    'connected_at': datetime.now().isoformat(),
                    'addr': addr
                }
                self.result.worker_hardware[worker_id] = hardware_info
                self.worker_update_count[worker_id] = 0
                self.worker_staleness_sum[worker_id] = 0
                self.worker_last_version[worker_id] = 0
                logger.info(f"Worker {worker_id} conectado | {hardware_info.get('cpu', 'Unknown')}")
            
            # Enviar configuracion
            await send_msg(writer, {
                'type': 'assign_id',
                'worker_id': worker_id,
                'total_workers': self.num_workers,
                'batch_size': self.batch_size,
                'dataset': self.dataset,
                'max_updates': self.max_updates,
                'model_version': self.model_version
            })
            
            # Esperar todos los workers para iniciar
            if len(self.workers) == self.num_workers:
                self.start_time = time.time()
                self.result.start_time = datetime.now().isoformat()
                logger.info("=" * 70)
                logger.info("TODOS LOS WORKERS CONECTADOS - MODO ASINCRONO")
                logger.info(f"   Training ID: {self.result.training_id}")
                logger.info(f"   Max updates: {self.max_updates}")
                logger.info(f"   Staleness penalty: {self.staleness_penalty}")
                logger.info(f"   Max staleness: {self.max_staleness}")
                logger.info("=" * 70)
                self.all_connected.set()
            else:
                remaining = self.num_workers - len(self.workers)
                logger.info(f"Esperando {remaining} worker(s)...")
                await self.all_connected.wait()
            
            # Enviar modelo inicial
            await self.send_model_to_worker(worker_id)
            
            # Bucle principal: recibir gradientes SIN BLOQUEAR
            while self.total_updates < self.max_updates and not self._training_done.is_set():
                try:
                    msg = await asyncio.wait_for(recv_msg(reader), timeout=30.0)
                    
                    if msg['type'] == 'gradient':
                        # Recibimos un gradiente - lo ponemos en la cola y seguimos
                        # NO bloqueamos al worker esperando a otros
                        grad_data = {
                            'gradients': msg['gradients'],
                            'worker_id': worker_id,
                            'model_version': msg.get('model_version', 0),
                            'loss': msg.get('loss', 0.0),
                            'timestamp': time.time()
                        }
                        
                        try:
                            self.gradient_queue.put_nowait(grad_data)
                            # Confirmar recepcion inmediata
                            await send_msg(writer, {
                                'type': 'gradient_received',
                                'queue_size': self.gradient_queue.qsize(),
                                'model_version': self.model_version
                            })
                        except asyncio.QueueFull:
                            logger.warning("Cola de gradientes llena - descartando gradiente")
                            await send_msg(writer, {
                                'type': 'gradient_rejected',
                                'reason': 'queue_full'
                            })
                    
                    elif msg['type'] == 'request_model':
                        # Worker pide modelo actualizado
                        await self.send_model_to_worker(worker_id)
                    
                    elif msg['type'] == 'done':
                        logger.info(f"Worker {worker_id} solicito finalizacion")
                        break
                        
                except asyncio.TimeoutError:
                    # Enviar heartbeat
                    try:
                        await send_msg(writer, {'type': 'heartbeat', 'model_version': self.model_version})
                    except:
                        break
                    continue
            
            # Training completado
            if self.total_updates >= self.max_updates and not self._training_done.is_set():
                logger.info("Max updates alcanzado!")
                self._training_done.set()
                await self.finish_training()
                await self.broadcast_training_done()
            
        except asyncio.IncompleteReadError:
            logger.warning(f"Worker {worker_id} desconectado")
        except Exception as e:
            logger.error(f"Error Worker {worker_id}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if worker_id is not None:
                async with self.lock:
                    if worker_id in self.workers:
                        del self.workers[worker_id]
            try:
                writer.close()
                await asyncio.wait_for(writer.wait_closed(), timeout=2.0)
            except:
                pass

    async def broadcast_training_done(self):
        msg = {'type': 'training_done'}
        tasks = []
        for _, (_, w) in list(self.workers.items()):
            try:
                tasks.append(send_msg(w, msg))
            except:
                pass
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def evaluate_and_log(self):
        """Evalua el modelo y guarda metricas."""
        try:
            accuracy = await self.evaluate_model()
            logger.info(f"Evaluacion update #{self.total_updates}: Accuracy = {accuracy:.2f}%")
            
            # Guardar en historial
            self.result.update_history.append({
                'update': self.total_updates,
                'eval_accuracy': round(accuracy, 2),
                'model_version': self.model_version,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.warning(f"Error en evaluacion: {e}")

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
                test_dataset, batch_size=64, shuffle=False, num_workers=0
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
            return accuracy
        except Exception as e:
            logger.warning(f"No se pudo evaluar: {e}")
            return 0.0

    async def finish_training(self):
        logger.info("FINALIZANDO ENTRENAMIENTO ASINCRONO...")
        
        end_time = time.time()
        total_seconds = end_time - self.start_time if self.start_time else 0
        
        final_accuracy = await self.evaluate_model()
        
        # Calcular loss promedio de los ultimos updates
        recent_losses = [h.get('loss', 0) for h in self.result.update_history[-20:] if 'loss' in h]
        final_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0.0
        
        self.result.end_time = datetime.now().isoformat()
        self.result.total_seconds = round(total_seconds, 2)
        self.result.total_updates = self.total_updates
        self.result.final_loss = round(final_loss, 6)
        self.result.final_accuracy = round(final_accuracy, 2)
        
        if self.update_times:
            self.result.avg_update_seconds = round(sum(self.update_times) / len(self.update_times), 3)
        
        for wid in self.worker_update_count:
            self.result.worker_updates[wid] = self.worker_update_count[wid]
            if self.worker_update_count[wid] > 0:
                self.result.worker_staleness_avg[wid] = round(
                    self.worker_staleness_sum[wid] / self.worker_update_count[wid], 2
                )
        
        self.result.print_summary()
        
        if self.save_results:
            self.result.save(self.results_dir)
        
        return self.result

    async def start(self):
        self.setup_signal_handlers()
        
        # Iniciar el procesador de gradientes en segundo plano
        processor_task = asyncio.create_task(self.gradient_processor_task())
        
        server = await asyncio.start_server(self.handle_worker, self.host, self.port)
        addrs = ', '.join(str(sock.getsockname()) for sock in server.sockets)
        
        logger.info("=" * 70)
        logger.info("SERVIDOR 100% ASINCRONO")
        logger.info(f"   Training ID: {self.result.training_id}")
        logger.info(f"   Checkpoints: {self.checkpoint_dir}")
        logger.info(f"   Escuchando en: {addrs}")
        logger.info(f"   Staleness penalty: {self.staleness_penalty}")
        logger.info(f"   Max staleness: {self.max_staleness}")
        logger.info("=" * 70)
        
        try:
            async with server:
                await server.serve_forever()
        except KeyboardInterrupt:
            logger.info("Interrupcion detectada")
            self.save_emergency_checkpoint()
            processor_task.cancel()
            try:
                await processor_task
            except asyncio.CancelledError:
                pass
            raise


# ==================== MAIN ====================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Servidor 100% Asincrono')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--num-classes', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'synthetic'])
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-updates', type=int, default=1000, 
                        help='Numero maximo de updates (en lugar de epocas)')
    parser.add_argument('--max-staleness', type=int, default=10,
                        help='Rechazar gradientes con staleness mayor a este valor')
    parser.add_argument('--staleness-penalty', default='linear',
                        choices=['linear', 'exponential', 'constant'],
                        help='Tipo de penalizacion por staleness')
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--results-dir', type=str)
    
    args = parser.parse_args()
    
    server = AsyncServer(
        host=args.host,
        port=args.port,
        num_workers=args.workers,
        num_classes=args.num_classes,
        lr=args.lr,
        dataset=args.dataset,
        batch_size=args.batch_size,
        max_updates=args.max_updates,
        max_staleness=args.max_staleness,
        staleness_penalty=args.staleness_penalty,
        save_results=not args.no_save,
        results_dir=args.results_dir
    )
    
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("Programa terminado")
