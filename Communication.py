# communication.py
import asyncio
import json
import struct
import numpy as np
from typing import Optional, Dict, Any

from Config import state, WorkerInfo

# ==============================
# Protocolo de comunicación
# ==============================
async def send_json(writer: asyncio.StreamWriter, data: dict):
    """Envía un mensaje JSON con prefijo de longitud."""
    message = json.dumps(data).encode()
    length = struct.pack(">I", len(message))
    writer.write(length + message)
    await writer.drain()

async def recv_json(reader: asyncio.StreamReader) -> Optional[dict]:
    """Recibe un mensaje JSON con prefijo de longitud."""
    try:
        raw_length = await reader.read(4)
        if not raw_length or len(raw_length) < 4:
            return None
        
        message_length = struct.unpack(">I", raw_length)[0]
        
        data = b""
        while len(data) < message_length:
            chunk_size = min(8192, message_length - len(data))
            packet = await reader.read(chunk_size)
            if not packet:
                return None
            data += packet
        
        return json.loads(data.decode())
    except Exception as e:
        print(f"[ERROR] recv_json: {e}")
        return None

# ==============================
# Manejador de workers
# ==============================
async def handle_worker(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, 
                        worker_id: int, dataset_chunk: np.ndarray):
    """Maneja la comunicación con un worker."""
    addr = writer.get_extra_info('peername')
    print(f"\n[Worker {worker_id}] Conectado desde {addr}")
    
    # Crear info del worker
    worker_info = WorkerInfo(
        writer=writer,
        reader=reader,
        worker_id=worker_id,
        dataset_chunk=dataset_chunk
    )
    state.workers[worker_id] = worker_info
    
    try:
        # 1. Enviar ID
        await send_json(writer, {"type": "worker_id", "worker_id": worker_id})
        
        # 2. Enviar chunk de datos
        await send_json(writer, {
            "type": "dataset_chunk",
            "indices": dataset_chunk.tolist()
        })
        
        # IMPORTANTE: Esperar a que TODOS los workers estén conectados
        # antes de comenzar el entrenamiento
        print(f"[Worker {worker_id}] Esperando a que todos los workers se conecten...")
        while len(state.workers) < state.expected_workers:
            await asyncio.sleep(0.5)
        
        # Ahora sí, comenzar el entrenamiento sincronizado
        for epoch in range(state.max_epochs):
            print(f"\n[Worker {worker_id}] === ÉPOCA {epoch + 1}/{state.max_epochs} ===")
            
            # ESPERAR INICIO DE ÉPOCA: Todos deben empezar la época con los MISMOS pesos
            # El servidor controla esto desde training_loop
            
            # Enviar pesos actuales (son los mismos para todos en esta época)
            await send_json(writer, {
                "type": "weights",
                "W1": state.global_weights["W1"].tolist(),
                "b1": state.global_weights["b1"].tolist(),
                "W2": state.global_weights["W2"].tolist(),
                "b2": state.global_weights["b2"].tolist(),
                "epoch": epoch + 1
            })
            
            # Esperar gradientes
            response = await recv_json(reader)
            
            if response is None or response.get("type") != "gradients":
                print(f"[Worker {worker_id}] ERROR: Conexión perdida")
                return
            
            # Guardar resultados
            worker_info.mark_epoch_done(response["grads"], response["loss"])
            
            # Registrar worker loss
            if worker_id not in state.worker_losses:
                state.worker_losses[worker_id] = []
            state.worker_losses[worker_id].append(response["loss"])
            
            print(f"[Worker {worker_id}] ✓ Loss: {response['loss']:.4f}")
            
            # IMPORTANTE: Marcar como listo y esperar a los demás
            state.mark_worker_ready(worker_id)
            
            # Si no soy el último, esperar
            if not state.all_workers_ready_for_epoch():
                print(f"[Worker {worker_id}] Esperando a otros workers...")
                await state.all_workers_ready.wait()
            
            print(f"[Worker {worker_id}] Continuando a siguiente época...")
        
        # 4. Fin de entrenamiento
        await send_json(writer, {
            "type": "training_complete",
            "message": "Entrenamiento finalizado"
        })
        print(f"[Worker {worker_id}] Entrenamiento completado")
        
    except Exception as e:
        print(f"[Worker {worker_id}] ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        writer.close()
        await writer.wait_closed()
        if worker_id in state.workers:
            del state.workers[worker_id]