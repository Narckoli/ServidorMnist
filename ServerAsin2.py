#!/usr/bin/env python3
"""
Servidor de entrenamiento distribuido asincrono para EfficientNetLite-0.
Mide tiempo, accuracy y guarda resultados en JSON.
"""

import asyncio
import struct
import pickle
import logging
import argparse
import random
import json
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("Server")


# ==================== RESULTADOS DEL ENTRENAMIENTO ====================
@dataclass
class TrainingResult:
    training_id: str = ""
    start_time: str = ""
    end_time: str = ""
    num_workers: int = 0
    num_classes: int = 10
    batch_size: int = 16
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
    batch_size: int = 16
    num_workers: int = 1
    chunk_size: int = 10
    seed: int = 42
    total_batches: int = field(init=False)
    total_chunks: int = field(init=False)
    chunks_per_worker: int = field(init=False)
    current_epoch: int = field(default=0)
    epoch_assignments: dict = field(default_factory=dict)
    
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
        self.epoch_assignments[epoch] = assignment
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


# ==================== SERVIDOR ====================
class AsyncDistServer:
    def __init__(self, host='0.0.0.0', port=5000, num_workers=1, 
                 num_classes=10, lr=0.001, dataset='cifar10',
                 batch_size=16, chunk_size=10, max_epochs=10,
                 save_results=True, results_dir=None):
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
        
        self.global_step = 0
        self.total_updates = 0
        self.current_epoch = 0
        
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
        self.epoch_start_times = {}
        self.epoch_losses = {}

    async def handle_worker(self, reader, writer):
        addr = writer.get_extra_info('peername')
        worker_id = None
        
        try:
            msg = await recv_msg(reader)
            if msg['type'] != 'handshake':
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
                self.result.worker_hardware[worker_id] = hardware_info
                logger.info(f"Worker {worker_id} conectado | {hardware_info.get('cpu', 'Unknown')}")
            
            await send_msg(writer, {
                'type': 'assign_id',
                'worker_id': worker_id,
                'total_workers': self.num_workers,
                'batch_size': self.batch_size,
                'chunk_size': self.dataset_manager.chunk_size,
                'total_samples': self.dataset_manager.total_samples,
                'dataset': self.dataset,
                'max_epochs': self.max_epochs
            })
            
            if len(self.workers) == self.num_workers:
                self.start_time = time.time()
                self.result.start_time = datetime.now().isoformat()
                logger.info("=" * 70)
                logger.info("TODOS LOS WORKERS CONECTADOS")
                logger.info(f"   Training ID: {self.result.training_id}")
                logger.info(f"   Epocas: {self.max_epochs}")
                logger.info(f"   Inicio: {self.result.start_time}")
                logger.info("=" * 70)
                self.all_connected.set()
            else:
                remaining = self.num_workers - len(self.workers)
                logger.info(f"Esperando {remaining} worker(s)...")
                await self.all_connected.wait()
            
            await self.broadcast_epoch_start(epoch=0)
            
            while True:
                msg = await recv_msg(reader)
                
                if msg['type'] == 'gradients':
                    worker_grads = msg['gradients']
                    step = msg['step']
                    loss = msg.get('loss', 0.0)
                    epoch = msg.get('epoch', 0)
                    
                    async with self.lock:
                        self.apply_gradients(worker_grads)
                        self.total_updates += 1
                        self.global_step = step
                        self.worker_info[worker_id]['updates_received'] += 1
                        
                        if epoch not in self.epoch_losses:
                            self.epoch_losses[epoch] = []
                        self.epoch_losses[epoch].append(loss)
                        
                        logger.info(f"W{worker_id} | Ep{epoch}/{self.max_epochs} | Loss:{loss:.4f} | Updates:{self.total_updates}")
                    
                    await self.send_model_to_worker(writer, epoch=epoch)
                    
                elif msg['type'] == 'epoch_complete':
                    completed_epoch = msg['epoch']
                    avg_loss = msg.get('avg_loss', 0.0)
                    
                    epoch_end_time = time.time()
                    if completed_epoch in self.epoch_start_times:
                        epoch_duration = epoch_end_time - self.epoch_start_times[completed_epoch]
                    else:
                        epoch_duration = 0.0
                    
                    self.result.epoch_history.append({
                        'epoch': completed_epoch,
                        'avg_loss': avg_loss,
                        'duration_seconds': round(epoch_duration, 2),
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    logger.info(f"Worker {worker_id} completo epoca {completed_epoch}/{self.max_epochs} ({epoch_duration:.1f}s)")
                    
                    async with self.lock:
                        self.worker_info[worker_id]['last_completed_epoch'] = completed_epoch
                        all_done = all(
                            self.worker_info.get(w, {}).get('last_completed_epoch', -1) >= completed_epoch
                            for w in self.workers
                        )
                    
                    if all_done and completed_epoch < self.max_epochs - 1:
                        next_epoch = completed_epoch + 1
                        logger.info(f"Iniciando epoca {next_epoch}/{self.max_epochs}")
                        await self.broadcast_epoch_start(epoch=next_epoch)
                        
                    elif all_done and completed_epoch >= self.max_epochs - 1:
                        await self.finish_training()
                        await self.broadcast_training_done()
                    
                elif msg['type'] == 'heartbeat':
                    await send_msg(writer, {'type': 'heartbeat_ack'})
                    
                elif msg['type'] == 'done':
                    logger.info(f"Worker {worker_id} finalizo")
                    break
                    
        except asyncio.IncompleteReadError:
            logger.warning(f"Worker {worker_id} desconectado")
        except Exception as e:
            logger.error(f"Error Worker {worker_id}: {e}")
        finally:
            if worker_id is not None:
                async with self.lock:
                    if worker_id in self.workers:
                        del self.workers[worker_id]
                    if worker_id in self.worker_info:
                        del self.worker_info[worker_id]
            writer.close()
            await writer.wait_closed()

    def apply_gradients(self, worker_grads):
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
        self.epoch_start_times[epoch] = time.time()
        self.current_epoch = epoch
        
        assignment = self.dataset_manager.generate_epoch_assignment(epoch)
        state_dict = self.model.state_dict()
        
        msg = {
            'type': 'epoch_start',
            'epoch': epoch,
            'state_dict': state_dict,
            'global_step': self.global_step,
            'chunk_assignment': assignment,
            'max_epochs': self.max_epochs
        }
        
        tasks = [send_msg(w, msg) for _, (_, w) in self.workers.items()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.info(f"Epoca {epoch}/{self.max_epochs} iniciada")

    async def broadcast_training_done(self):
        msg = {
            'type': 'training_done',
            'total_epochs': self.max_epochs,
            'total_updates': self.total_updates
        }
        tasks = [send_msg(w, msg) for _, (_, w) in self.workers.items()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def finish_training(self):
        logger.info("FINALIZANDO ENTRENAMIENTO...")
        
        end_time = time.time()
        total_seconds = end_time - self.start_time if self.start_time else 0
        
        final_loss = 0.0
        if self.max_epochs - 1 in self.epoch_losses and self.epoch_losses[self.max_epochs - 1]:
            final_loss = sum(self.epoch_losses[self.max_epochs - 1]) / len(self.epoch_losses[self.max_epochs - 1])
        elif self.epoch_losses:
            last_epoch = max(self.epoch_losses.keys())
            final_loss = sum(self.epoch_losses[last_epoch]) / len(self.epoch_losses[last_epoch])
        
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
            logger.info(f"Resultados guardados en: {filepath}")
        
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
            logger.info(f"Accuracy en test set: {accuracy:.2f}% ({correct}/{total})")
            return accuracy
            
        except Exception as e:
            logger.warning(f"No se pudo evaluar accuracy: {e}")
            return 0.0

    async def send_model_to_worker(self, writer, epoch: int = 0):
        state_dict = self.model.state_dict()
        await send_msg(writer, {
            'type': 'model_update',
            'state_dict': state_dict,
            'global_step': self.global_step,
            'epoch': epoch,
            'max_epochs': self.max_epochs
        })

    async def start(self):
        server = await asyncio.start_server(self.handle_worker, self.host, self.port)
        addrs = ', '.join(str(sock.getsockname()) for sock in server.sockets)
        
        logger.info("=" * 70)
        logger.info("SERVIDOR DE ENTRENAMIENTO DISTRIBUIDO")
        logger.info(f"   Training ID: {self.result.training_id}")
        logger.info(f"   Modelo: EfficientNetLite-0")
        logger.info(f"   Dataset: {self.dataset.upper()}")
        logger.info(f"   Workers: {self.num_workers}")
        logger.info(f"   Epocas: {self.max_epochs}")
        logger.info(f"   Resultados en: {self.results_dir}")
        logger.info(f"   Escuchando en: {addrs}")
        logger.info("=" * 70)
        
        async with server:
            await server.serve_forever()


# ==================== MAIN ====================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Servidor de entrenamiento distribuido')
    parser.add_argument('--host', default='0.0.0.0', help='Host')
    parser.add_argument('--port', type=int, default=5000, help='Puerto')
    parser.add_argument('--workers', type=int, default=1, help='Workers')
    parser.add_argument('--num-classes', type=int, default=10, help='Clases')
    parser.add_argument('--lr', type=float, default=0.001, help='LR')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'synthetic'])
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--chunk-size', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=1, help='EPOCAS TOTALES')
    parser.add_argument('--no-save', action='store_true', help='No guardar resultados')
    parser.add_argument('--results-dir', type=str, help='Directorio para resultados')
    
    args = parser.parse_args()
    
    server = AsyncDistServer(
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
        results_dir=args.results_dir
    )
    
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("Servidor detenido")
        if server.save_results and server.start_time:
            asyncio.run(server.finish_training())