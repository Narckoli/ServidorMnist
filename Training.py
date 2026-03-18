# training.py
import asyncio
import time
import numpy as np

from Config import state
from Model import evaluate_model, average_gradients, apply_gradients
from Metrics import plot_metrics

async def training_loop(state):
    """Loop principal con sincronización correcta."""
    state.training_start_time = time.time()
    print_interval = 50
    
    print(f"\n{'='*60}")
    print("INICIO DE ENTRENAMIENTO")
    print(f"{'='*60}")
    
    # Asegurar que todos los workers están conectados
    print("Verificando conexión de todos los workers...")
    while len(state.workers) < state.expected_workers:
        await asyncio.sleep(0.5)
    print(f"✓ {len(state.workers)} workers conectados")
    
    for epoch in range(state.max_epochs):
        start_time = time.time()
        state.current_epoch = epoch
        
        # PREPARAR NUEVA ÉPOCA
        state.reset_epoch_sync()
        
        should_print = (epoch + 1) % print_interval == 0 or epoch == 0 or epoch == state.max_epochs - 1
        
        if should_print:
            print(f"\n{'='*60}")
            print(f'ÉPOCA {epoch + 1}/{state.max_epochs}')
            print(f"{'='*60}")
            print("Enviando pesos a todos los workers...")
        
        # Enviar pesos a TODOS los workers (esto ya lo hace cada worker handler)
        # Pero esperamos a que TODOS terminen
        
        # ESPERAR a que todos los workers completen la época
        await state.all_workers_ready.wait()
        
        # Recolectar gradientes de TODOS los workers
        all_grads = []
        worker_losses = []
        
        for wid, worker in state.workers.items():
            if worker.epoch_completed.is_set():
                all_grads.append(worker.current_grads)
                worker_losses.append(worker.current_loss)
        
        # Verificar que tenemos todos los gradientes
        if len(all_grads) != state.expected_workers:
            print(f"ERROR: Solo tenemos {len(all_grads)} de {state.expected_workers} gradientes")
            continue
        
        # Calcular promedio
        avg_train_loss = np.mean(worker_losses)
        avg_grads = average_gradients(all_grads, state.global_weights)
        
        # ACTUALIZAR PESOS GLOBALES (solo después de que TODOS terminaron)
        state.global_weights = apply_gradients(
            state.global_weights, avg_grads, state.learning_rate
        )
        
        # Evaluar
        test_loss, test_accuracy = evaluate_model(
            state.X_test, state.y_test, state.global_weights
        )
        
        # Guardar métricas
        state.train_losses.append(avg_train_loss)
        state.test_losses.append(test_loss)
        state.test_accuracies.append(test_accuracy)
        
        epoch_time = time.time() - start_time
        state.epoch_times.append(epoch_time)
        
        if should_print:
            print(f"✓ Train Loss: {avg_train_loss:.4f}")
            print(f"✓ Test Loss:  {test_loss:.4f}")
            print(f"✓ Test Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
            print(f"⏱️  Tiempo de época: {epoch_time:.2f}s")
        
        # Pequeña pausa para evitar saturación
        await asyncio.sleep(0.1)
    
    print(f"\n{'='*60}")
    print("ENTRENAMIENTO COMPLETADO")
    print(f"{'='*60}")

def mark_worker_ready(self, worker_id: int):
    """Marca un worker como listo para esta época."""
    if worker_id in self.workers_ready_for_epoch:
        self.workers_ready_for_epoch[worker_id] = True
    
    # Verificar si todos están listos
    if all(self.workers_ready_for_epoch.values()):
        self.all_workers_ready.set()
        if hasattr(self, 'current_epoch'):
            print(f"\n🎯 ¡Todos los workers completaron la época {self.current_epoch + 1}!")

def all_workers_ready_for_epoch(self) -> bool:
    """Verifica si todos los workers están listos."""
    return len(self.workers_ready_for_epoch) == self.expected_workers and \
           all(self.workers_ready_for_epoch.values())