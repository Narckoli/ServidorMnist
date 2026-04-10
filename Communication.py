# Communication.py - Servidor
# Protocolo: JSON con delimitador \n en AMBOS sentidos (servidor y worker)
import json
import asyncio
import struct
import numpy as np
from Config import state

# ─────────────────────────────────────────────
# Primitivas de bajo nivel
# ─────────────────────────────────────────────

async def send_json(writer, data: dict):
    """Envía un dict como JSON terminado en \\n."""
    line = json.dumps(data) + '\n'
    writer.write(line.encode())
    await writer.drain()

async def recv_json(reader) -> dict | None:
    """Lee una línea y la deserializa como JSON. Retorna None si la conexión se cerró."""
    try:
        line = await reader.readline()
        if not line:
            return None
        return json.loads(line.decode())
    except (json.JSONDecodeError, asyncio.IncompleteReadError):
        return None

# ─────────────────────────────────────────────
# Mensajes de setup
# ─────────────────────────────────────────────

async def send_worker_id(writer, worker_id: int):
    """Paso 1: envía el ID al worker."""
    await send_json(writer, {"type": "id", "worker_id": worker_id})

async def send_dataset_info(writer, dataset_name: str, input_size: int, model_type: str):
    """Paso 2: envía metadatos del dataset y modelo."""
    await send_json(writer, {
        "type": "dataset_info",
        "dataset": dataset_name,          # worker lee msg["dataset"]
        "input_size": input_size,
        "model_type": model_type,
    })

async def send_chunk_indices(writer, chunk: np.ndarray):
    """
    Paso 3: envía los índices del chunk como JSON (lista de ints).
    Evitamos el protocolo binario para mantener un único protocolo en toda la sesión.
    """
    await send_json(writer, {
        "type": "chunk",
        "indices": chunk.tolist(),
    })

# ─────────────────────────────────────────────
# Handler principal de cada worker
# ─────────────────────────────────────────────

async def handle_worker(reader, writer, worker_id: int, chunk: np.ndarray):
    """
    Gestiona el ciclo completo de un worker:
      setup (3 pasos) → esperar "ready" → bucle de entrenamiento.
    """
    try:
        # ── Registrar conexión ──────────────────────────────────────
        async with state.lock:
            state.worker_readers[worker_id] = reader
            state.worker_writers[worker_id] = writer
            state.worker_chunks[worker_id]  = chunk

        # ── Paso 1: ID ──────────────────────────────────────────────
        await send_worker_id(writer, worker_id)
        print(f"[Worker {worker_id}] ✓ ID enviado")

        # ── Paso 2: Dataset info ────────────────────────────────────
        await send_dataset_info(writer, state.dataset_name,
                                state.input_size, state.model_type)
        print(f"[Worker {worker_id}] ✓ Dataset info enviada")

        # ── Paso 3: Chunk de índices ────────────────────────────────
        await send_chunk_indices(writer, chunk)
        print(f"[Worker {worker_id}] ✓ Chunk enviado ({len(chunk)} muestras)")

        # ── Esperar "ready" del worker ──────────────────────────────
        msg = await recv_json(reader)
        if msg and msg.get("type") == "ready":
            async with state.lock:
                state.mark_worker_ready(worker_id)
            print(f"[Worker {worker_id}] ✓ Listo para entrenar")
        else:
            print(f"[Worker {worker_id}] ✗ Respuesta inesperada en setup: {msg}")
            return

        # ── Bucle de entrenamiento ──────────────────────────────────
        while state.training_active:
            try:
                msg = await asyncio.wait_for(recv_json(reader), timeout=600.0)
                if msg is None:
                    print(f"[Worker {worker_id}] Conexión cerrada")
                    break

                mtype = msg.get("type")

                if mtype == "gradients":
                    grads = msg.get("gradients")   # worker envía "gradients"
                    loss  = msg.get("loss")
                    epoch = msg.get("epoch")

                    async with state.lock:
                        state.worker_gradients[worker_id] = grads
                        state.worker_losses[worker_id]    = loss

                    print(f"[Worker {worker_id}] Gradientes recibidos "
                          f"(época {epoch}, loss: {loss:.4f})")

                    # Si todos los workers respondieron, liberar el evento
                    async with state.lock:
                        if len(state.worker_gradients) == len(state.worker_writers):
                            print(f"[Training] Todos los gradientes recibidos — época {epoch}")
                            state.all_workers_ready.set()

                elif mtype == "ping":
                    await send_json(writer, {"type": "pong"})

            except asyncio.TimeoutError:
                # Heartbeat
                try:
                    await send_json(writer, {"type": "ping"})
                except Exception:
                    break

    except (ConnectionError, asyncio.CancelledError) as e:
        print(f"[Worker {worker_id}] Conexión perdida: {e}")
    except Exception as e:
        print(f"[Worker {worker_id}] Error inesperado: {e}")
        import traceback; traceback.print_exc()
    finally:
        async with state.lock:
            state.worker_writers.pop(worker_id, None)
            state.worker_readers.pop(worker_id, None)
            state.worker_chunks.pop(worker_id, None)
            state.worker_ready.pop(worker_id, None)
            state.workers_ready.pop(worker_id, None)
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass
        print(f"[Worker {worker_id}] Conexión cerrada limpiamente")