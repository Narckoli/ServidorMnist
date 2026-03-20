# servidor/Communication.py
import asyncio
import json
import struct
import numpy as np
from typing import Optional

from Config import state, WorkerInfo

async def send_json(writer: asyncio.StreamWriter, data: dict):
    """Envía datos JSON con prefijo de longitud."""
    message = json.dumps(data).encode()
    length = struct.pack(">I", len(message))
    writer.write(length + message)
    await writer.drain()

async def recv_json(reader: asyncio.StreamReader) -> Optional[dict]:
    """Recibe datos JSON con prefijo de longitud."""
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

async def handle_worker(reader, writer, worker_id, dataset_chunk):
    """Maneja la comunicación con un worker - VERSIÓN CORREGIDA."""
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
        print(f"[Worker {worker_id}] ✓ ID enviado")
        
        # 2. Enviar chunk de datos
        await send_json(writer, {
            "type": "dataset_chunk", 
            "indices": dataset_chunk.tolist()
        })
        print(f"[Worker {worker_id}] ✓ Chunk enviado ({len(dataset_chunk)} muestras)")
        
        # 3. IMPORTANTE: Esperar a que el worker confirme que está listo
        ready_msg = await recv_json(reader)
        if not ready_msg or ready_msg.get("type") != "worker_ready":
            print(f"[Worker {worker_id}] ❌ No se recibió confirmación de ready")
            return
        
        print(f"[Worker {worker_id}] ✓ Worker listo para entrenar")
        worker_info.ready_for_training = True
        
        # 4. BUCLE DE ENTRENAMIENTO: El servidor inicia cada época
        for epoch in range(state.max_epochs):
            print(f"\n[Worker {worker_id}] 📋 ÉPOCA {epoch + 1}/{state.max_epochs}")
            
            # ESPERAR a que TODOS los workers estén listos para esta época
            # (esto lo maneja el training_loop)
            
            # El servidor ENVÍA los pesos primero
            await send_json(writer, {
                "type": "weights",
                "W1": state.global_weights["W1"].tolist(),
                "b1": state.global_weights["b1"].tolist(),
                "W2": state.global_weights["W2"].tolist(),
                "b2": state.global_weights["b2"].tolist(),
                "epoch": epoch + 1
            })
            print(f"[Worker {worker_id}] ✓ Pesos enviados")
            
            # ESPERAR los gradientes del worker
            response = await recv_json(reader)
            
            if response is None or response.get("type") != "gradients":
                print(f"[Worker {worker_id}] ❌ Error: No se recibieron gradientes")
                return
            
            # Guardar resultados
            worker_info.mark_epoch_done(response["grads"], response["loss"])
            
            if worker_id not in state.worker_losses:
                state.worker_losses[worker_id] = []
            state.worker_losses[worker_id].append(response["loss"])
            
            print(f"[Worker {worker_id}] ✓ Loss recibida: {response['loss']:.4f}")
            
            # Marcar como listo para sincronización
            state.mark_worker_ready(worker_id)
            
            # Esperar a que TODOS los workers terminen esta época
            if not state.all_workers_ready_for_epoch():
                print(f"[Worker {worker_id}] ⏳ Esperando otros workers...")
                await state.all_workers_ready.wait()
            
            print(f"[Worker {worker_id}] ➡️ Continuando...")
        
        # 5. Fin de entrenamiento
        await send_json(writer, {
            "type": "training_complete",
            "message": "Entrenamiento finalizado"
        })
        print(f"[Worker {worker_id}] ✓ Entrenamiento completado")
        
    except Exception as e:
        print(f"[Worker {worker_id}] ❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        writer.close()
        await writer.wait_closed()
        if worker_id in state.workers:
            del state.workers[worker_id]