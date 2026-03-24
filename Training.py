# training.py
import asyncio
import time
import numpy as np
from Config import state
from Model import average_gradients, apply_gradients, evaluate_model
from Export_Metrics import save_final_summary  

async def training_loop():
    """Bucle principal de entrenamiento."""
    print("\n" + "="*70)
    print("INICIO DE ENTRENAMIENTO")
    print("="*70)
    
    # Inicializar estructuras para sincronización
    async with state.lock:
        state.worker_gradients = {}
        state.worker_losses = {}
        state.all_workers_ready = asyncio.Event()
        state.current_epoch = 0
        state.training_start_time = time.time()
        state.epoch_events = {}
    
    print(" Esperando que todos los workers completen su setup...")
    
    # Esperar a que todos los workers estén listos
    while True:
        async with state.lock:
            workers_ready = state.check_all_workers_ready_for_training()
        if workers_ready:
            break
        await asyncio.sleep(0.5)
    
    print("✓ Todos los workers están listos!\n")
    
    # Evaluación inicial
    print("Evaluación inicial del modelo global:")
    initial_loss, initial_acc = evaluate_model(state.X_test, state.y_test, state.global_weights)
    print(f"  Loss: {initial_loss:.4f}, Accuracy: {initial_acc:.4f}\n")
    
    for epoch in range(state.max_epochs):
        async with state.lock:
            state.current_epoch = epoch
        print("="*70)
        print(f"ÉPOCA {epoch + 1}/{state.max_epochs}")
        print("="*70)
        
        async with state.lock:
            state.epoch_start_time = time.time()
        
        # Limpiar estructuras de la época
        async with state.lock:
            state.worker_gradients.clear()
            state.worker_losses.clear()
            state.all_workers_ready.clear()
            
            epoch_event = asyncio.Event()
            state.epoch_events[epoch] = epoch_event
        
        # Enviar pesos a todos los workers
        print(f"\n[Época {epoch + 1}] Enviando pesos a los workers...")
        
        async with state.lock:
            workers_to_send = list(state.worker_writers.items())
        
        for worker_id, writer in workers_to_send:
            try:
                await send_weights_to_worker(writer, state.global_weights, epoch + 1)
                print(f"   Pesos enviados al Worker {worker_id}")
            except Exception as e:
                print(f"   Error con Worker {worker_id}: {e}")
        
        print(f"\n[Época {epoch + 1}] Esperando gradientes de los workers...")
        
        try:
            async with state.lock:
                ready_event = state.all_workers_ready
            await asyncio.wait_for(ready_event.wait(), timeout=60.0)
            
            print("\n[Época {}] Promediando gradientes...".format(epoch + 1))
            
            async with state.lock:
                gradients_list = list(state.worker_gradients.values())
                losses_list = list(state.worker_losses.values())
                worker_losses_dict = dict(state.worker_losses)  # Copiar pérdidas por worker
            
            avg_gradients = average_gradients(gradients_list, state.global_weights)
            
            # Aplicar gradientes
            state.global_weights = apply_gradients(
                state.global_weights,
                avg_gradients,
                state.learning_rate
            )
            
            # Calcular pérdida promedio
            avg_loss = np.mean(losses_list)
            
            # Evaluar modelo
            test_loss, test_acc = evaluate_model(
                state.X_test, state.y_test, state.global_weights
            )
            
            # Calcular tiempo de época
            async with state.lock:
                epoch_time = time.time() - state.epoch_start_time
            
            print(f"\n RESULTADOS ÉPOCA {epoch + 1}:")
            print(f"  Pérdida promedio (workers): {avg_loss:.4f}")
            print(f"  Pérdida (test): {test_loss:.4f}")
            print(f"  Accuracy (test): {test_acc:.4f}")
            print(f"  Tiempo: {epoch_time:.2f} segundos")
            
        except asyncio.TimeoutError:
            print(f" Timeout: No se recibieron gradientes de todos los workers")
            async with state.lock:
                received = len(state.worker_gradients)
                expected = len(state.worker_writers)
            print(f"  Workers que respondieron: {received}/{expected}")
            break
            
        except Exception as e:
            print(f" Error en época {epoch + 1}: {e}")
            import traceback
            traceback.print_exc()
            break
        
        print()
    
    # Evaluación final
    print("\n" + "="*70)
    print("ENTRENAMIENTO COMPLETADO")
    print("="*70)
    final_loss, final_acc = evaluate_model(state.X_test, state.y_test, state.global_weights)
    print(f"Modelo final - Loss: {final_loss:.4f}, Accuracy: {final_acc:.4f}")
    
    total_time = time.time() - state.training_start_time
    print(f"Tiempo total: {total_time:.2f} segundos")
    print("="*70)
    
    # ========== GUARDAR RESUMEN FINAL ==========
    save_final_summary(final_loss, final_acc, total_time)
    # ===========================================

async def send_weights_to_worker(writer, weights, epoch):
    """Envía pesos a un worker usando JSON."""
    from Communication import send_json
    await send_json(writer, {
        "type": "weights",
        "W1": weights["W1"].tolist(),
        "b1": weights["b1"].tolist(),
        "W2": weights["W2"].tolist(),
        "b2": weights["b2"].tolist(),
        "epoch": epoch
    })