# Training.py - Servidor
import asyncio
import time
import numpy as np

from Config import state
from Communication import send_json
from Model import average_gradients, apply_gradients, evaluate_model
from Export_Metrics import save_epoch_metrics, save_final_summary


async def send_weights_to_worker(writer, weights: dict, epoch: int):
    """
    Envía los pesos globales al worker.
    Formato: {"type":"weights", "epoch":N, "model_type":"mlp"|"cnn", "weights":{...}}
    Todos los valores numpy se convierten a lista para ser serializables en JSON.
    """
    serializable = {
        k: (v.tolist() if isinstance(v, np.ndarray) else v)
        for k, v in weights.items()
    }
    await send_json(writer, {
        "type":       "weights",
        "epoch":      epoch,
        "model_type": state.model_type,
        "weights":    serializable,   # worker accede a msg["weights"]
    })


async def training_loop():
    """Bucle principal de entrenamiento federado."""
    print("\n" + "=" * 70)
    print("INICIO DE ENTRENAMIENTO")
    print("=" * 70)

    # ── Inicializar estructuras de sincronización ──────────────────────────────
    async with state.lock:
        state.worker_gradients   = {}
        state.worker_losses      = {}
        state.all_workers_ready  = asyncio.Event()
        state.current_epoch      = 0
        state.training_start_time = time.time()
        state.epoch_events       = {}

    # ── Esperar setup de todos los workers ────────────────────────────────────
    print("Esperando que todos los workers completen su setup...")
    while True:
        async with state.lock:
            ready = state.check_all_workers_ready_for_training()
        if ready:
            break
        await asyncio.sleep(0.5)
    print("✓ Todos los workers listos\n")

    # ── Evaluación inicial ────────────────────────────────────────────────────
    if state.model_type == 'cnn' and state.model:
        init_loss, init_acc = state.model.evaluate(state.X_test, state.y_test)
    else:
        init_loss, init_acc = evaluate_model(
            state.X_test, state.y_test, state.global_weights)
    print(f"Evaluación inicial — Loss: {init_loss:.4f}, Accuracy: {init_acc:.4f}\n")

    # ── Bucle por épocas ──────────────────────────────────────────────────────
    for epoch in range(state.max_epochs):
        async with state.lock:
            state.current_epoch    = epoch
            state.epoch_start_time = time.time()
            state.worker_gradients.clear()
            state.worker_losses.clear()
            state.all_workers_ready.clear()

        print("=" * 70)
        print(f"ÉPOCA {epoch + 1}/{state.max_epochs}")
        print("=" * 70)

        # ── Enviar pesos a todos los workers ──────────────────────────────────
        async with state.lock:
            workers_snapshot = list(state.worker_writers.items())

        print(f"\n[Época {epoch+1}] Enviando pesos...")
        for wid, writer in workers_snapshot:
            try:
                await send_weights_to_worker(writer, state.global_weights, epoch + 1)
                print(f"   Pesos → Worker {wid}")
            except Exception as e:
                print(f"   ✗ Error enviando a Worker {wid}: {e}")

        # ── Esperar gradientes ────────────────────────────────────────────────
        print(f"\n[Época {epoch+1}] Esperando gradientes...")
        try:
            async with state.lock:
                ready_event = state.all_workers_ready

            await asyncio.wait_for(ready_event.wait(), timeout=300.0)

            async with state.lock:
                gradients_list = list(state.worker_gradients.values())
                losses_list    = list(state.worker_losses.values())
                worker_losses_snapshot = dict(state.worker_losses)

            # ── Agregar gradientes y actualizar pesos ─────────────────────────
            if state.model_type == 'cnn' and state.model:
                avg_grads = state.model.average_gradients(gradients_list)
                state.global_weights = state.model.apply_gradients(
                    state.global_weights, avg_grads, state.learning_rate)
            else:
                avg_grads = average_gradients(gradients_list, state.global_weights)
                state.global_weights = apply_gradients(
                    state.global_weights, avg_grads, state.learning_rate)

            avg_loss = float(np.mean(losses_list))

            # ── Evaluar modelo actualizado ────────────────────────────────────
            if state.model_type == 'cnn' and state.model:
                test_loss, test_acc = state.model.evaluate(state.X_test, state.y_test)
            else:
                test_loss, test_acc = evaluate_model(
                    state.X_test, state.y_test, state.global_weights)

            async with state.lock:
                epoch_time = time.time() - state.epoch_start_time

            # ── Guardar métricas en state ─────────────────────────────────────
            state.train_losses.append(avg_loss)
            state.test_losses.append(test_loss)
            state.test_accuracies.append(test_acc)
            state.epoch_times.append(epoch_time)

            # Historial por worker
            async with state.lock:
                for wid, loss_val in worker_losses_snapshot.items():
                    state.worker_loss_history.setdefault(wid, []).append(loss_val)

            # ── Exportar métricas de la época ─────────────────────────────────
            save_epoch_metrics(epoch, avg_loss, test_loss, test_acc,
                               epoch_time, worker_losses_snapshot)

            print(f"\n RESULTADOS ÉPOCA {epoch + 1}:")
            print(f"  Train loss (avg workers): {avg_loss:.4f}")
            print(f"  Test  loss:               {test_loss:.4f}")
            print(f"  Test  accuracy:           {test_acc:.4f}")
            print(f"  Tiempo época:             {epoch_time:.2f}s")

        except asyncio.TimeoutError:
            async with state.lock:
                received = len(state.worker_gradients)
                expected = len(state.worker_writers)
                missing  = set(state.worker_writers.keys()) - set(state.worker_gradients.keys())
            print(f"\n✗ Timeout época {epoch+1}: {received}/{expected} workers respondieron")
            print(f"  Workers faltantes: {missing}")
            break

        except Exception as e:
            print(f"\n✗ Error en época {epoch+1}: {e}")
            import traceback; traceback.print_exc()
            break

        print()

    # ── Evaluación final ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 70)

    if state.model_type == 'cnn' and state.model:
        final_loss, final_acc = state.model.evaluate(state.X_test, state.y_test)
    else:
        final_loss, final_acc = evaluate_model(
            state.X_test, state.y_test, state.global_weights)

    total_time = time.time() - state.training_start_time
    print(f"Loss final: {final_loss:.4f} | Accuracy final: {final_acc:.4f}")
    print(f"Tiempo total: {total_time:.2f}s")
    print("=" * 70)

    save_final_summary(final_loss, final_acc, total_time)