
import os
os.makedirs('/mnt/agents/output/efficientnet_dist_v2', exist_ok=True)

# ============================================================
# SERVIDOR V2 - Con mezcla de batches y partición inteligente
# ============================================================
"""
Servidor de entrenamiento distribuido asíncrono para EfficientNetLite-0.

Diseño para hardware:
- Primarios: AMD Ryzen 3 Serie 7000 (4C/8T, buena eficiencia energética)
- Secundarios: Intel Core i9 10th Gen (10C/20T, alto rendimiento cuando disponible)

Características:
- Define N-Workers por consola
- Asigna ID único + N-Workers a cada worker
- Implementa mezcla de batches (shuffling distribuido) para que todos los
  workers vean TODO el dataset progresivamente, no solo su partición fija
- Soporta workers heterogéneos (Ryzen 3 vs Core i9)
"""

import asyncio
import json
import struct
import pickle
import logging
import argparse
import hashlib
import random
from datetime import datetime
from collections import defaultdict, deque
from dataclasses import dataclass, field

import torch
import torch.nn as nn

# ==================== CONFIGURACIÓN DE LOGGING ====================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("Server")


# ==================== MODELO EFFICIENTNET LITE-0 (OPTIMIZADO CPU) ====================
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
        self.stride = stride
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.use_residual = (stride == 1 and in_ch == out_ch)
        hidden_dim = in_ch * expand_ratio

        layers = []
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_ch, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                Swish()
            ]
        layers += [
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size//2, 
                     groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            Swish()
        ]
        if se_ratio is not None:
            layers.append(SEBlock(hidden_dim, se_ratio))
        layers += [
            nn.Conv2d(hidden_dim, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        ]
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
    """
    EfficientNet-Lite-0 optimizado para CPU.
    
    Lite-0 es la versión más pequeña de la familia EfficientNet-Lite,
    ideal para CPUs como Ryzen 3 7000 series (4C/8T).
    """
    
    CONFIG = [
        # (expand_ratio, out_ch, kernel_size, stride, se_ratio, num_repeat)
        (1, 16, 3, 1, None, 1),
        (6, 24, 3, 2, None, 2),
        (6, 40, 5, 2, None, 2),
        (6, 80, 3, 2, 0.25, 3),
        (6, 112, 5, 1, 0.25, 3),
        (6, 192, 5, 2, 0.25, 4),
        (6, 320, 3, 1, None, 1),
    ]

    def __init__(self, num_classes=10, width_mult=1.0, depth_mult=1.0, dropout_rate=0.2):
        super().__init__()
        out_ch = self._round_filters(32, width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(3, out_ch, 3, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            Swish()
        )
        
        blocks = []
        in_ch = out_ch
        for expand_ratio, out_ch_cfg, kernel_size, stride, se_ratio, num_repeat in self.CONFIG:
            out_ch = self._round_filters(out_ch_cfg, width_mult)
            num_repeat = self._round_repeats(num_repeat, depth_mult)
            for i in range(num_repeat):
                s = stride if i == 0 else 1
                blocks.append(MBConvBlock(in_ch, out_ch, expand_ratio, kernel_size, s, se_ratio, dropout_rate))
                in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        
        head_ch = self._round_filters(1280, width_mult)
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, head_ch, 1, bias=False),
            nn.BatchNorm2d(head_ch),
            Swish(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(head_ch, num_classes)
        )

    def _round_filters(self, filters, mult):
        from math import ceil
        return int(ceil(filters * mult / 8) * 8)

    def _round_repeats(self, repeats, mult):
        from math import ceil
        return int(ceil(repeats * mult))

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ==================== GESTOR DE DATASET DISTRIBUIDO ====================
@dataclass
class DistributedDatasetManager:
    """
    Gestiona la distribución de batches entre workers con mezcla progresiva.
    
    PROBLEMA que resuelve:
    - Si cada worker tiene una partición FIJA del dataset, nunca ve los datos
      de los otros workers → el modelo no generaliza bien.
    
    SOLUCIÓN (mezcla de batches):
    - El dataset se divide en "chunks" (grupos de batches)
    - Cada época, los chunks se reasignan aleatoriamente entre workers
    - Así, en la época 1 el worker 0 ve chunks [0,2,4] y en la época 2 ve [1,3,5]
    - TODOS los workers ven TODO el dataset, solo en diferente orden y momento
    
    CIFAR-10: 50,000 imágenes de entrenamiento
    - Batch size 32 → ~1,563 batches totales
    - Con 2 workers y chunk_size=10 batches → ~156 chunks por worker por época
    """
    
    total_samples: int = 50000          # CIFAR-10 train
    batch_size: int = 32
    num_workers: int = 2
    chunk_size: int = 10                # batches por chunk
    seed: int = 42
    
    total_batches: int = field(init=False)
    total_chunks: int = field(init=False)
    chunks_per_worker: int = field(init=False)
    current_epoch: int = field(default=0)
    epoch_assignments: dict = field(default_factory=dict)  # epoch -> {worker_id: [chunk_indices]}
    
    def __post_init__(self):
        self.total_batches = (self.total_samples + self.batch_size - 1) // self.batch_size
        self.total_chunks = (self.total_batches + self.chunk_size - 1) // self.chunk_size
        self.chunks_per_worker = self.total_chunks // self.num_workers
        logger.info(f"📊 DatasetManager: {self.total_batches} batches → "
                    f"{self.total_chunks} chunks (chunk_size={self.chunk_size}) → "
                    f"~{self.chunks_per_worker} chunks/worker/época")
    
    def generate_epoch_assignment(self, epoch: int) -> dict:
        """
        Genera la asignación de chunks para una época específica.
        
        Algoritmo:
        1. Crea lista de todos los chunk indices [0, 1, 2, ..., total_chunks-1]
        2. Mezcla con seed = epoch + base_seed (determinístico pero diferente por época)
        3. Reparte equitativamente entre workers
        
        Returns:
            {worker_id: [chunk_idx, chunk_idx, ...], ...}
        """
        rng = random.Random(self.seed + epoch)
        all_chunks = list(range(self.total_chunks))
        rng.shuffle(all_chunks)
        
        assignment = {}
        base = self.chunks_per_worker
        
        for w in range(self.num_workers):
            start = w * base
            end = start + base if w < self.num_workers - 1 else len(all_chunks)
            assignment[w] = sorted(all_chunks[start:end])
        
        self.epoch_assignments[epoch] = assignment
        logger.info(f"🎲 Época {epoch}: chunks asignados → "
                    f"{ {k: len(v) for k, v in assignment.items()} }")
        return assignment
    
    def get_worker_chunks_for_epoch(self, worker_id: int, epoch: int) -> list:
        """Devuelve los chunks que le tocan a un worker en una época."""
        if epoch not in self.epoch_assignments:
            self.generate_epoch_assignment(epoch)
        return self.epoch_assignments[epoch].get(worker_id, [])
    
    def get_batch_indices_for_chunk(self, chunk_idx: int) -> tuple:
        """
        Convierte un chunk_idx a los índices de batch que contiene.
        
        chunk_idx=0 → batches [0, 1, 2, ..., chunk_size-1]
        chunk_idx=1 → batches [chunk_size, chunk_size+1, ...]
        """
        start_batch = chunk_idx * self.chunk_size
        end_batch = min(start_batch + self.chunk_size, self.total_batches)
        
        # Convertir índices de batch a índices de muestra
        start_sample = start_batch * self.batch_size
        end_sample = min(end_batch * self.batch_size, self.total_samples)
        
        return list(range(start_sample, end_sample))
    
    def get_worker_sample_indices_for_epoch(self, worker_id: int, epoch: int) -> list:
        """
        Devuelve TODOS los índices de muestra que un worker debe usar en una época.
        """
        chunks = self.get_worker_chunks_for_epoch(worker_id, epoch)
        indices = []
        for chunk_idx in chunks:
            indices.extend(self.get_batch_indices_for_chunk(chunk_idx))
        return indices


# ==================== PROTOCOLO DE COMUNICACIÓN ====================
async def send_msg(writer, data):
    """Envía datos serializados precedidos por su longitud (4 bytes, big-endian)."""
    payload = pickle.dumps(data)
    length = struct.pack('>I', len(payload))
    writer.write(length + payload)
    await writer.drain()


async def recv_msg(reader):
    """Recibe datos serializados con prefijo de longitud."""
    length_data = await reader.readexactly(4)
    length = struct.unpack('>I', length_data)[0]
    payload = await reader.readexactly(length)
    return pickle.loads(payload)


# ==================== SERVIDOR ====================
class AsyncDistServer:
    def __init__(self, host='0.0.0.0', port=5000, num_workers=1, 
                 num_classes=10, lr=0.001, dataset='cifar10',
                 batch_size=32, chunk_size=10):
        self.host = host
        self.port = port
        self.num_workers = num_workers
        self.lr = lr
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Modelo global
        self.model = EfficientNetLite0(num_classes=num_classes)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        
        # Gestor de dataset distribuido con mezcla de batches
        if dataset == 'cifar10':
            total_samples = 50000
        else:
            total_samples = 10000  # synthetic
            
        self.dataset_manager = DistributedDatasetManager(
            total_samples=total_samples,
            batch_size=batch_size,
            num_workers=num_workers,
            chunk_size=chunk_size,
            seed=42
        )
        
        # Estado de workers
        self.workers = {}          # worker_id -> (reader, writer)
        self.worker_info = {}      # worker_id -> {hardware, connected_at, ...}
        self.lock = asyncio.Lock()
        self.all_connected = asyncio.Event()
        
        self.global_step = 0
        self.total_updates = 0
        self.current_epoch = 0

    async def handle_worker(self, reader, writer):
        """Maneja la conexión de un worker."""
        addr = writer.get_extra_info('peername')
        worker_id = None
        
        try:
            # 1. Recibir handshake con info del hardware
            msg = await recv_msg(reader)
            if msg['type'] != 'handshake':
                logger.warning(f"[{addr}] Handshake inválido")
                return
            
            hardware_info = msg.get('hardware', 'unknown')
            
            async with self.lock:
                worker_id = len(self.workers)
                self.workers[worker_id] = (reader, writer)
                self.worker_info[worker_id] = {
                    'hardware': hardware_info,
                    'connected_at': datetime.now().isoformat(),
                    'addr': addr,
                    'updates_received': 0
                }
                logger.info(f"🟢 Worker {worker_id} conectado desde {addr}")
                logger.info(f"   Hardware: {hardware_info}")
                logger.info(f"   Progreso: {len(self.workers)}/{self.num_workers} workers")
            
            # 2. Enviar ID asignado + configuración de dataset
            await send_msg(writer, {
                'type': 'assign_id',
                'worker_id': worker_id,
                'total_workers': self.num_workers,
                'batch_size': self.batch_size,
                'chunk_size': self.dataset_manager.chunk_size,
                'total_samples': self.dataset_manager.total_samples,
                'dataset': self.dataset
            })
            
            # 3. Esperar a que todos los workers se conecten
            if len(self.workers) == self.num_workers:
                logger.info("=" * 60)
                logger.info("✅ TODOS LOS WORKERS CONECTADOS")
                logger.info(f"   Workers: {list(self.workers.keys())}")
                logger.info(f"   Hardware detectado: {[self.worker_info[w]['hardware'] for w in self.workers]}")
                logger.info("   Iniciando entrenamiento distribuido asíncrono...")
                logger.info("=" * 60)
                self.all_connected.set()
            else:
                remaining = self.num_workers - len(self.workers)
                logger.info(f"⏳ Esperando {remaining} worker(s) más...")
                await self.all_connected.wait()
            
            # 4. Difundir modelo inicial + asignación de chunks para época 0
            await self.broadcast_epoch_start(epoch=0)
            
            # 5. Bucle principal: recibir gradientes, actualizar modelo, enviar modelo
            while True:
                msg = await recv_msg(reader)
                
                if msg['type'] == 'gradients':
                    worker_grads = msg['gradients']
                    step = msg['step']
                    loss = msg.get('loss', 0.0)
                    epoch = msg.get('epoch', 0)
                    
                    async with self.lock:
                        # Aplicar gradientes al modelo global
                        self.apply_gradients(worker_grads)
                        self.total_updates += 1
                        self.global_step = step
                        self.worker_info[worker_id]['updates_received'] += 1
                        
                        logger.info(f"📥 Gradiente de Worker {worker_id} | "
                                    f"Epoch {epoch} | Step: {step} | Loss: {loss:.4f} | "
                                    f"Updates globales: {self.total_updates}")
                    
                    # Enviar modelo actualizado de vuelta
                    await self.send_model_to_worker(writer, epoch=epoch)
                    
                elif msg['type'] == 'epoch_complete':
                    completed_epoch = msg['epoch']
                    logger.info(f"🎯 Worker {worker_id} completó época {completed_epoch}")
                    
                    # Verificar si TODOS los workers completaron esta época
                    async with self.lock:
                        self.worker_info[worker_id]['last_completed_epoch'] = completed_epoch
                        
                        all_done = all(
                            self.worker_info.get(w, {}).get('last_completed_epoch', -1) >= completed_epoch
                            for w in self.workers
                        )
                    
                    if all_done:
                        # Todos completaron → nueva época con mezcla diferente
                        next_epoch = completed_epoch + 1
                        logger.info(f"🎲 TODOS completaron época {completed_epoch}. "
                                    f"Generando asignación para época {next_epoch}...")
                        await self.broadcast_epoch_start(epoch=next_epoch)
                    
                elif msg['type'] == 'heartbeat':
                    await send_msg(writer, {'type': 'heartbeat_ack'})
                    
                elif msg['type'] == 'done':
                    logger.info(f"🏁 Worker {worker_id} finalizó entrenamiento")
                    break
                    
        except asyncio.IncompleteReadError:
            logger.warning(f"⚠️ Worker {worker_id} desconectado inesperadamente")
        except Exception as e:
            logger.error(f"❌ Error con Worker {worker_id}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if worker_id is not None:
                async with self.lock:
                    if worker_id in self.workers:
                        del self.workers[worker_id]
                    if worker_id in self.worker_info:
                        del self.worker_info[worker_id]
                logger.info(f"🔴 Worker {worker_id} desconectado. Quedan: {len(self.workers)}")
            writer.close()
            await writer.wait_closed()

    def apply_gradients(self, worker_grads):
        """Aplica gradientes recibidos al modelo global (optimización asíncrona)."""
        self.optimizer.zero_grad()
        
        for name, param in self.model.named_parameters():
            if name in worker_grads:
                grad_tensor = worker_grads[name]
                if param.grad is None:
                    param.grad = grad_tensor.clone()
                else:
                    param.grad += grad_tensor
        
        self.optimizer.step()

    async def broadcast_epoch_start(self, epoch: int):
        """
        Difunde a todos los workers:
        1. El modelo global actual
        2. La asignación de chunks para la nueva época (mezcla)
        """
        # Generar asignación para esta época
        assignment = self.dataset_manager.generate_epoch_assignment(epoch)
        state_dict = self.model.state_dict()
        
        msg = {
            'type': 'epoch_start',
            'epoch': epoch,
            'state_dict': state_dict,
            'global_step': self.global_step,
            'chunk_assignment': assignment  # {worker_id: [chunk_indices]}
        }
        
        tasks = []
        for wid, (r, w) in self.workers.items():
            tasks.append(send_msg(w, msg))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.info(f"📤 Época {epoch} iniciada. Modelo + asignación de chunks enviados a "
                        f"{len(self.workers)} workers")

    async def send_model_to_worker(self, writer, epoch: int = 0):
        """Envía el modelo global actual a un worker específico."""
        state_dict = self.model.state_dict()
        await send_msg(writer, {
            'type': 'model_update',
            'state_dict': state_dict,
            'global_step': self.global_step,
            'epoch': epoch
        })

    async def start(self):
        """Inicia el servidor."""
        server = await asyncio.start_server(
            self.handle_worker, self.host, self.port
        )
        
        addrs = ', '.join(str(sock.getsockname()) for sock in server.sockets)
        logger.info("=" * 60)
        logger.info("🚀 SERVIDOR DE ENTRENAMIENTO DISTRIBUIDO ASÍNCRONO")
        logger.info(f"   Modelo: EfficientNetLite-0")
        logger.info(f"   Dataset: {self.dataset.upper()}")
        logger.info(f"   Workers esperados: {self.num_workers}")
        logger.info(f"   Batch size: {self.batch_size}")
        logger.info(f"   Chunk size: {self.dataset_manager.chunk_size}")
        logger.info(f"   Escuchando en: {addrs}")
        logger.info("=" * 60)
        logger.info(f"⏳ Esperando {self.num_workers} workers...")
        
        async with server:
            await server.serve_forever()


# ==================== MAIN ====================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Servidor de entrenamiento distribuido asíncrono - EfficientNetLite-0'
    )
    parser.add_argument('--host', default='0.0.0.0', help='Host del servidor')
    parser.add_argument('--port', type=int, default=5000, help='Puerto del servidor')
    parser.add_argument('--workers', type=int, default=2, 
                        help='NÚMERO DE WORKERS ESPERADOS (configurable por consola)')
    parser.add_argument('--num-classes', type=int, default=10, help='Número de clases')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'synthetic'],
                        help='Dataset a usar')
    parser.add_argument('--batch-size', type=int, default=32, help='Tamaño de batch')
    parser.add_argument('--chunk-size', type=int, default=10, 
                        help='Batches por chunk (para mezcla de batches)')
    
    args = parser.parse_args()
    
    server = AsyncDistServer(
        host=args.host,
        port=args.port,
        num_workers=args.workers,
        num_classes=args.num_classes,
        lr=args.lr,
        dataset=args.dataset,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size
    )
    
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("🛑 Servidor detenido por el usuario")