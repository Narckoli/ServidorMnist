# training.py
import asyncio
import time
import numpy as np

from Config import state
from Model import evaluate_model, average_gradients, apply_gradients
from Metrics import plot_metrics

async def training_loop():
    """Loop principal de entrenamiento."""
    state.training_start_time = time.time()
    print_interval = 50
    
    print(f"\n{'='*60}")
    print("INICIO DE ENTRENAMIENTO")
    print(f"{'='*60}")
    
    for epoch in range(state.max_epochs):
        start_time = time.time()
        state.current_epoch = epoch
        
        should_print = (epoch + 1) % print_interval == 0 or epoch == 0 or epoch == state.max_epochs - 1
        
        if should_print:
            print(f"\n{'='*60}")
            print(f'ÉPOCA {epoch + 1}/{state.max_epochs}')
            print(f"{'='*60}")
            print("Esperando gradientes de todos los workers...")
        
        # Reset para nueva época
        state.all_workers_ready.clear()
        for worker in state.workers.values():
            worker.reset_for_next_epoch()
        
        # Esperar a que todos los workers completen
        await state.all_workers_ready.wait()
        
        # Recolectar gradientes
        all_grads, worker_losses = state.get_completed_workers_data()
        
        if not all_grads:
            print("ERROR: No hay gradientes para procesar")
            break
        
        # Calcular métricas
        avg_train_loss = np.mean(worker_losses)
        
        # Promediar gradientes
        avg_grads = average_gradients(all_grads, state.global_weights)
        
        # Actualizar pesos globales
        state.global_weights = apply_gradients(
            state.global_weights, avg_grads, state.learning_rate
        )
        
        # Evaluar en test set
        test_loss, test_accuracy = evaluate_model(
            state.X_test, state.y_test, state.global_weights
        )
        
        # Guardar métricas
        state.train_losses.append(avg_train_loss)
        state.test_losses.append(test_loss)
        state.test_accuracies.append(test_accuracy)
        
        epoch_time = time.time() - start_time
        state.epoch_times.append(epoch_time)
        
        # Mostrar progreso
        if should_print:
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Test Loss:  {test_loss:.4f}")
            print(f"Test Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
            print(f"Tiempo de época: {epoch_time:.2f}s")
        else:
            print(".", end="", flush=True)
            if (epoch + 1) % 10 == 0:
                print(f" {epoch + 1}")
        
        # Pequeña pausa para evitar saturación
        await asyncio.sleep(0.1)
    
    print(f"\n{'='*60}")
    print("ENTRENAMIENTO COMPLETADO")
    print(f"{'='*60}")
    
    # Mostrar gráficas
    plot_metrics()