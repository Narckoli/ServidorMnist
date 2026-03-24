# servidor/communication.py
import asyncio
import json
import struct
import numpy as np
from typing import Optional

from Config import state 

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
    
    # Registrar el worker
    state.worker_writers[worker_id] = writer
    state.worker_readers[worker_id] = reader
    state.worker_chunks[worker_id] = dataset_chunk
    
    try:
        # 1. Enviar ID
        await send_json(writer, {"type": "worker_id", "worker_id": worker_id})
        print(f"[Worker {worker_id}] ✓ ID enviado")
        
        # 2. Enviar información del dataset
        await send_json(writer, {
            "type": "dataset_info",
            "dataset_name": state.dataset_name,
            "input_size": state.input_size
        })
        print(f"[Worker {worker_id}] ✓ Dataset info enviada: {state.dataset_name} ({state.input_size} features)")
        
        # 3. Enviar chunk de datos
        await send_json(writer, {
            "type": "dataset_chunk",
            "indices": dataset_chunk.tolist()
        })
        print(f"[Worker {worker_id}] ✓ Chunk enviado ({len(dataset_chunk)} muestras)")
        
        # 4. Esperar confirmación de READY
        ready_msg = await recv_json(reader)
        if ready_msg and ready_msg.get("type") == "worker_ready":
            state.mark_worker_ready(worker_id)
            print(f"[Worker {worker_id}] ✓ Worker listo para entrenar")
        else:
            print(f"[Worker {worker_id}] ✗ No se recibió confirmación de ready")
            return
        
        # 5. Bucle principal de comunicación - MANEJA TODOS LOS MENSAJES AQUÍ
        while True:
            # Esperar mensaje del worker
            msg = await recv_json(reader)
            
            if msg is None:
                print(f"[Worker {worker_id}] Conexión perdida")
                break
            
            msg_type = msg.get("type")
            
            if msg_type == "gradients":
                # Recibir gradientes del worker
                grads = {
                    "W1": np.array(msg["grads"]["W1"]),
                    "b1": np.array(msg["grads"]["b1"]),
                    "W2": np.array(msg["grads"]["W2"]),
                    "b2": np.array(msg["grads"]["b2"])
                }
                loss = msg.get("loss", 0.0)
                epoch = msg.get("epoch", 0)
                
                print(f"[Worker {worker_id}] Gradientes recibidos (época {epoch}, loss: {loss:.4f})")
                
                # Guardar gradientes para el entrenamiento
                state.worker_gradients[worker_id] = grads
                state.worker_losses[worker_id] = loss
                
                # Verificar si todos los workers han enviado sus gradientes
                if len(state.worker_gradients) == len(state.worker_writers):
                    print(f"[Training] Todos los gradientes recibidos para época {epoch}")
                    state.all_workers_ready.set()
                
            elif msg_type == "training_complete":
                print(f"[Worker {worker_id}] Entrenamiento completado")
                break
                
            else:
                print(f"[Worker {worker_id}] Mensaje desconocido: {msg_type}")
        
    except Exception as e:
        print(f"[Worker {worker_id}] ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Eliminar worker de los diccionarios de manera segura
        async with state.lock:
            state.worker_writers.pop(worker_id, None)
            state.worker_readers.pop(worker_id, None)
        
        try:
            writer.close()
            await writer.wait_closed()
            print(f"[Worker {worker_id}] Conexión cerrada")
        except:
            pass