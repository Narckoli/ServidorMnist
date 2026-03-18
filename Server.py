# server.py
import asyncio
import time

from Config import state
from Dataset import load_mnist_dataset, stratified_split
from Model import init_weights
from Communication import handle_worker
from Training import training_loop

async def setup_configuration():
    """Configuración interactiva del entrenamiento."""
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
    
    print(f"\nConfiguración: {state.expected_workers} workers, {state.max_epochs} épocas, lr={state.learning_rate}")

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
    # Cargar dataset
    X_train, y_train, state.X_test, state.y_test = load_mnist_dataset()
    
    # Configuración interactiva
    await setup_configuration()
    
    # Inicializar pesos globales
    state.global_weights = init_weights()
    print("Pesos inicializados (He initialization)")
    
    # Preparar chunks para workers
    dataset_chunks = stratified_split(y_train, state.expected_workers)
    
    # Contador de workers conectados
    connected_count = 0
    
    async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Manejador de conexiones de clientes."""
        nonlocal connected_count
        connected_count += 1
        worker_id = connected_count
        chunk = dataset_chunks[worker_id - 1]
        
        print(f"Worker {worker_id} conectado ({connected_count}/{state.expected_workers})")
        await handle_worker(reader, writer, worker_id, chunk)
        
        # Verificar si se perdió un worker durante el entrenamiento
        if len(state.workers) < state.expected_workers and state.current_epoch < state.max_epochs:
            print(f"ADVERTENCIA: Worker {worker_id} desconectado durante entrenamiento")
    
    # Iniciar servidor
    server = await asyncio.start_server(handle_client, state.HOST, state.PORT)
    print(f"\nServidor iniciado en {state.HOST}:{state.PORT}")
    print(f"Esperando {state.expected_workers} workers...")
    
    # Esperar a que todos se conecten
    while connected_count < state.expected_workers:
        await asyncio.sleep(0.1)
    
    print(f"\n✓ Todos los workers conectados. Iniciando entrenamiento...")
    
    # Ejecutar entrenamiento
    await training_loop()
    
    # Cerrar servidor
    server.close()
    await server.wait_closed()
    
    # Mostrar tiempo total
    print_total_time()
    print("\nServidor finalizado")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nServidor interrumpido por el usuario")