# servidor/server.py
import asyncio
import time

from Config import state
from Dataset import load_dataset_by_name, stratified_split
from Model import init_weights
from Communication import handle_worker
from Training import training_loop

async def setup_configuration():
    """Configuración interactiva del entrenamiento."""
    # Selección del dataset
    print("\n=== SELECCIÓN DE DATASET ===")
    print("1. MNIST (28x28 imágenes, 784 características)")
    print("2. CIFAR-10 (32x32x3 imágenes, 3072 características)")
    
    while True:
        try:
            choice = input("\nSelecciona el dataset (1 o 2): ").strip()
            if choice == '1':
                state.dataset_name = 'mnist'
                break
            elif choice == '2':
                state.dataset_name = 'cifar10'
                break
            else:
                print("Opción no válida. Por favor selecciona 1 o 2.")
        except:
            pass
    
    # Número de workers
    while True:
        try:
            state.expected_workers = int(input("\n¿Cuántos workers se conectarán? "))
            if state.expected_workers > 0:
                break
        except ValueError:
            pass
        print("Por favor ingresa un número válido")
    
    # Épocas
    try:
        custom_epochs = input(f"¿Número de épocas? (default: {state.max_epochs}): ").strip()
        if custom_epochs:
            state.max_epochs = int(custom_epochs)
    except:
        pass
    
    # Learning rate
    try:
        custom_lr = input(f"¿Learning rate? (default: {state.learning_rate}): ").strip()
        if custom_lr:
            state.learning_rate = float(custom_lr)
    except:
        pass
    
    print(f"\nConfiguración: Dataset={state.dataset_name.upper()}, {state.expected_workers} workers, {state.max_epochs} épocas, lr={state.learning_rate}")

def print_total_time():
    """Muestra el tiempo total de entrenamiento."""
    if state.training_start_time:
        total_elapsed = time.time() - state.training_start_time
        total_ms = total_elapsed * 1000
        mins = int(total_elapsed // 60)
        secs = int(total_elapsed % 60)
        ms = int((total_elapsed % 1) * 1000)
        
        print(f"\n{'='*70}")
        print("ENTRENAMIENTO COMPLETADO - TIEMPO TOTAL")
        print(f"{'='*70}")
        print(f"  Milisegundos: {total_ms:,.0f} ms")
        print(f"  Formateado:   {mins:02d} min {secs:02d} s {ms:03d} ms")
        print(f"  Total:        {total_elapsed:.3f} segundos")
        print(f"{'='*70}")

async def main():
    """Punto de entrada principal del servidor."""
    server = None
    
    try:
        # Configuración interactiva
        await setup_configuration()
        
        # Cargar dataset seleccionado
        X_train, y_train, state.X_test, state.y_test = load_dataset_by_name(state.dataset_name)
        
        # Configurar input size en el modelo
        state.input_size = X_train.shape[1]
        
        # Inicializar pesos globales con el input size correcto
        state.global_weights = init_weights(input_size=state.input_size)
        print(f"Pesos inicializados (He initialization) para {state.input_size} características de entrada")
        
        # Preparar chunks para workers
        dataset_chunks = stratified_split(y_train, state.expected_workers)
        
        # Contador de workers conectados
        connected_count = 0
        all_workers_connected = asyncio.Event()
        
        async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
            nonlocal connected_count
            connected_count += 1
            worker_id = connected_count
            chunk = dataset_chunks[worker_id - 1]
            
            print(f"✓ Worker {worker_id} conectado ({connected_count}/{state.expected_workers})")
            
            if connected_count == state.expected_workers:
                print("\n ¡Todos los workers conectados!")
                all_workers_connected.set()
            
            try:
                await handle_worker(reader, writer, worker_id, chunk)
            except Exception as e:
                print(f"Error en worker {worker_id}: {e}")
            finally:
                # Asegurarse de que el worker se elimina del diccionario cuando se desconecta
                async with state.lock:
                    state.worker_writers.pop(worker_id, None)
                    state.worker_readers.pop(worker_id, None)
        
        # Iniciar servidor
        server = await asyncio.start_server(handle_client, state.HOST, state.PORT)
        print(f"\n Servidor iniciado en {state.HOST}:{state.PORT}")
        print(f" Esperando {state.expected_workers} workers...")
        
        # Esperar a que todos los workers se conecten
        await all_workers_connected.wait()
        
        # Pequeña pausa para asegurar que todos los workers están listos
        print(" Esperando configuración de workers...")
        await asyncio.sleep(2)
        
        # Iniciar entrenamiento
        await training_loop()
        
        print("✅ Entrenamiento completado exitosamente")
        
    except KeyboardInterrupt:
        print("\n\n Servidor interrumpido por el usuario")
    except Exception as e:
        print(f"\n Error en servidor: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n=== INICIANDO LIMPIEZA DEL SERVIDOR ===")
        
        # Cerrar todas las conexiones de workers - USAR COPIA DEL DICCIONARIO
        # Para evitar el error de modificación durante iteración, hacemos una copia
        workers_to_close = []
        async with state.lock:
            workers_to_close = list(state.worker_writers.items())
        
        for worker_id, writer in workers_to_close:
            try:
                writer.close()
                await writer.wait_closed()
                print(f"✓ Conexión con Worker {worker_id} cerrada")
            except Exception as e:
                print(f"Error cerrando Worker {worker_id}: {e}")
        
        # Limpiar diccionarios después de cerrar conexiones
        async with state.lock:
            state.worker_writers.clear()
            state.worker_readers.clear()
        
        # Cerrar servidor
        if server is not None:
            print("Cerrando servidor...")
            server.close()
            await server.wait_closed()
            print("✓ Servidor cerrado correctamente")
        
        # Mostrar tiempo total
        print_total_time()
        print("\n Servidor finalizado")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n Servidor interrumpido por el usuario")