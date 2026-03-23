# servidor/communication.py
import asyncio
import json
import struct
import numpy as np
from typing import Optional

from Config import state, WorkerInfo

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

async def handle_worker(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, 
                        worker_id: int, dataset_chunk: np.ndarray):
    """Maneja la comunicación con un worker."""
    addr = writer.get_extra_info('peername')
    print(f"\n[Worker {worker_id}] Conectado desde {addr}")
    
    # REGISTRAR EL WRITER EN worker_writers
    state.worker_writers[worker_id] = writer  # <--- AGREGAR ESTO
    
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
        print(f"[Worker {worker_id}] ✓ ID enviado")
        
        # 2. Enviar chunk de datos
        await send_json(writer, {
            "type": "dataset_chunk",
            "indices": dataset_chunk.tolist()
        })
        print(f"[Worker {worker_id}] ✓ Chunk enviado ({len(dataset_chunk)} muestras)")
        
        # 3. Esperar confirmación de READY
        ready_msg = await recv_json(reader)
        if ready_msg and ready_msg.get("type") == "worker_ready":
            worker_info.mark_ready()
            print(f"[Worker {worker_id}]  Worker listo para entrenar")
        else:
            print(f"[Worker {worker_id}]  No se recibió confirmación de ready")
            return
        
        # 4. Esperar a que TODOS los workers estén listos
        print(f"[Worker {worker_id}]  Esperando a que todos los workers estén listos...")
        while not state.check_all_workers_ready_for_training():
            await asyncio.sleep(0.5)
        
        # 5. Bucle de épocas - AHORA SINCRONIZADO
        for epoch in range(state.max_epochs):
            # Verificar si estamos en la época correcta
            while state.current_epoch != epoch:
                await asyncio.sleep(0.1)
            
            print(f"\n[Worker {worker_id}]  ÉPOCA {epoch + 1}/{state.max_epochs}")
            
            # Enviar pesos actuales
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
                print(f"[Worker {worker_id}]  ERROR: Conexión perdida")
                return
            
            # Guardar resultados
            worker_info.mark_epoch_done(response["grads"], response["loss"])
            
            # Guardar loss del worker
            if worker_id not in state.worker_losses:
                state.worker_losses[worker_id] = []
            state.worker_losses[worker_id].append(response["loss"])
            
            print(f"[Worker {worker_id}]  Loss recibida: {response['loss']:.4f}")
            
            # Notificar al coordinador
            if state.check_all_workers_ready():
                state.all_workers_ready.set()
            
            # Esperar a que el coordinador procese esta época
            await state.all_workers_ready.wait()
            
            # Pequeña pausa antes de la siguiente época
            await asyncio.sleep(0.4)
            
        # 6. Fin de entrenamiento
        await send_json(writer, {
            "type": "training_complete",
            "message": "Entrenamiento finalizado"
        })
        print(f"[Worker {worker_id}]  Entrenamiento completado")
        
    except Exception as e:
        print(f"[Worker {worker_id}]  ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # LIMPIAR AMBOS DICCIONARIOS
        if worker_id in state.worker_writers:
            del state.worker_writers[worker_id]
        if worker_id in state.workers:
            del state.workers[worker_id]
        
        try:
            writer.close()
            await writer.wait_closed()
        except:
            pass
            
        print(f"[Worker {worker_id}]  Conexión cerrada")