# Server.py
import asyncio
import time

from Config import state
from Dataset import load_dataset_by_name, stratified_split
from Model_Cnn import create_cnn_model
from Communication import handle_worker
from Training import training_loop

# 100 MB — suficiente para cualquier chunk de índices JSON
_BUFFER_LIMIT = 100 * 1024 * 1024


async def setup_configuration():
    print("\n=== SELECCIÓN DE MODELO ===")
    print("1. MLP (Multilayer Perceptron) - Simple, rápido")
    print("2. CNN (Convolutional Neural Network) - Mejor para CIFAR-10")

    while True:
        try:
            choice = input("\nSelecciona el modelo (1 o 2): ").strip()
            if choice == '1':
                state.model_type = 'mlp'
                print("✅ Usando modelo MLP")
                break
            elif choice == '2':
                state.model_type = 'cnn'
                print("✅ Usando modelo CNN")
                break
            else:
                print("Opción no válida.")
        except Exception:
            pass

    print("\n=== SELECCIÓN DE DATASET ===")
    print("1. MNIST  (28x28, 784 características)")
    print("2. CIFAR-10 (32x32x3, 3072 características)")

    while True:
        try:
            choice = input("\nSelecciona el dataset (1 o 2): ").strip()
            if choice == '1':
                state.dataset_name = 'mnist'
                if state.model_type == 'cnn':
                    print("⚠️  CNN en MNIST funciona, pero MLP es suficiente")
                break
            elif choice == '2':
                state.dataset_name = 'cifar10'
                if state.model_type == 'mlp':
                    print("⚠️  MLP no es óptimo para CIFAR-10, considera CNN")
                break
            else:
                print("Opción no válida.")
        except Exception:
            pass

    while True:
        try:
            state.expected_workers = int(input("\n¿Cuántos workers se conectarán? "))
            if state.expected_workers > 0:
                break
        except ValueError:
            pass
        print("Por favor ingresa un número válido")

    try:
        v = input(f"¿Número de épocas? (default: {state.max_epochs}): ").strip()
        if v:
            state.max_epochs = int(v)
    except Exception:
        pass

    try:
        v = input(f"¿Learning rate? (default: {state.learning_rate}): ").strip()
        if v:
            state.learning_rate = float(v)
    except Exception:
        pass

    print(f"\nConfiguración: Modelo={state.model_type.upper()}, "
          f"Dataset={state.dataset_name.upper()}, "
          f"{state.expected_workers} workers, "
          f"{state.max_epochs} épocas, lr={state.learning_rate}")


def print_total_time():
    if state.training_start_time:
        t = time.time() - state.training_start_time
        mins, secs = int(t // 60), int(t % 60)
        ms = int((t % 1) * 1000)
        print(f"\n{'='*70}")
        print("ENTRENAMIENTO COMPLETADO - TIEMPO TOTAL")
        print(f"{'='*70}")
        print(f"  {mins:02d} min {secs:02d} s {ms:03d} ms  ({t:.3f} s)")
        print(f"{'='*70}")


async def main():
    server = None
    try:
        await setup_configuration()

        X_train, y_train, state.X_test, state.y_test = \
            load_dataset_by_name(state.dataset_name)
        state.input_size = X_train.shape[1]

        if state.model_type == 'cnn':
            state.model = create_cnn_model()
            state.global_weights = state.model.init_weights(
                input_size=state.input_size)
            print(f"✅ Modelo CNN inicializado para {state.dataset_name.upper()}")
        else:
            from Model import init_weights
            state.global_weights = init_weights(input_size=state.input_size)
            print(f"✅ Modelo MLP inicializado ({state.input_size} características)")
            state.model = None

        dataset_chunks = stratified_split(y_train, state.expected_workers)

        connected_count = 0
        all_workers_connected = asyncio.Event()

        async def handle_client(reader: asyncio.StreamReader,
                                writer: asyncio.StreamWriter):
            nonlocal connected_count
            connected_count += 1
            worker_id = connected_count
            chunk = dataset_chunks[worker_id - 1]

            print(f"✓ Worker {worker_id} conectado "
                  f"({connected_count}/{state.expected_workers})")

            if connected_count == state.expected_workers:
                print("\n¡Todos los workers conectados!")
                all_workers_connected.set()

            try:
                await handle_worker(reader, writer, worker_id, chunk)
            except Exception as e:
                print(f"Error en worker {worker_id}: {e}")
            finally:
                async with state.lock:
                    state.worker_writers.pop(worker_id, None)
                    state.worker_readers.pop(worker_id, None)

        # ── limit=_BUFFER_LIMIT es el fix clave ──────────────────────────────
        server = await asyncio.start_server(
            handle_client, state.HOST, state.PORT,
            limit=_BUFFER_LIMIT,
        )
        print(f"\nServidor iniciado en {state.HOST}:{state.PORT}")
        print(f"Esperando {state.expected_workers} workers...")

        await all_workers_connected.wait()
        print("Esperando configuración de workers...")
        await asyncio.sleep(2)

        await training_loop()
        print("Entrenamiento completado exitosamente")

    except KeyboardInterrupt:
        print("\n\nServidor interrumpido por el usuario")
    except Exception as e:
        print(f"\nError en servidor: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n=== LIMPIEZA DEL SERVIDOR ===")
        async with state.lock:
            workers_to_close = list(state.worker_writers.items())

        for wid, writer in workers_to_close:
            try:
                writer.close()
                await writer.wait_closed()
                print(f"✓ Worker {wid} cerrado")
            except Exception as e:
                print(f"Error cerrando Worker {wid}: {e}")

        async with state.lock:
            state.worker_writers.clear()
            state.worker_readers.clear()

        if server is not None:
            server.close()
            await server.wait_closed()
            print("✓ Servidor cerrado")

        print_total_time()
        print("\nServidor finalizado")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nServidor interrumpido por el usuario")