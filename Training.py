# Training.py - Servidor

import asyncio
import time
import numpy as np

from Config import state
from Communication import send_json
from Export_Metrics import save_epoch_metrics, save_final_summary

# ── Hiperparámetros ───────────────────────────────────────────────────────────
_CNN_LR_MAX     = 0.005   # lr >= este valor con CNN → se ajusta automáticamente
_CNN_LR_DEFAULT = 0.0003
_MOMENTUM       = 0.5
_GRAD_NORM_CLIP = 5.0
# ─────────────────────────────────────────────────────────────────────────────


def _load_weights_into_model(weights: dict):
    """
    Carga el state_dict en el modelo del servidor respetando los dtypes
    originales (float32 para pesos, int64 para num_batches_tracked).
    """
    import torch
    current_sd = state.model.model.state_dict()
    new_sd = {
        k: torch.tensor(np.array(v), dtype=current_sd[k].dtype)
        for k, v in weights.items()
    }
    state.model.model.load_state_dict(new_sd, strict=True)


def _evaluate():
    """
    Evalúa el modelo global.

    Para CNN: evalúa en modo TRAIN con torch.no_grad().
    Esto hace que BatchNorm use las estadísticas del batch actual
    en lugar de running_mean/running_var, que en el servidor nunca
    se actualizan porque el servidor no hace forward pass con datos
    de entrenamiento.
    """
    if state.model_type == 'cnn' and state.model:
        import torch
        _load_weights_into_model(state.global_weights)

        model = state.model.model
        model.train()   # BatchNorm usa stats del batch, no running stats

        X = state.X_test
        y = state.y_test

        if X.shape[1] == 3072:
            X = X.reshape(-1, 3, 32, 32)

        X_t = torch.tensor(X, dtype=torch.float32).to(state.model.device)
        y_t = torch.tensor(y, dtype=torch.long).to(state.model.device)

        # Procesar en batches para no agotar RAM
        batch_size  = 256
        total_loss  = 0.0
        correct     = 0
        n           = len(X_t)

        with torch.no_grad():
            for start in range(0, n, batch_size):
                end    = min(start + batch_size, n)
                xb, yb = X_t[start:end], y_t[start:end]
                out    = model(xb)
                import torch.nn.functional as F
                total_loss += F.cross_entropy(out, yb).item() * (end - start)
                correct    += (out.argmax(1) == yb).sum().item()

        return total_loss / n, correct / n

    else:
        from Model import evaluate_model
        return evaluate_model(state.X_test, state.y_test, state.global_weights)


def _average_gradients(gradients_list: list) -> dict:
    if not gradients_list:
        return {}
    avg = {}
    for key in gradients_list[0]:
        avg[key] = np.mean([np.array(g[key]) for g in gradients_list], axis=0)
    return avg


def _average_bn_buffers(bn_buffers_list: list) -> dict:
    """
    Promedia los buffers de BatchNorm (running_mean, running_var)
    enviados por los workers. num_batches_tracked se suma (no promedia).
    """
    if not bn_buffers_list:
        return {}
    avg = {}
    for key in bn_buffers_list[0]:
        arrays = [np.array(b[key]) for b in bn_buffers_list]
        if 'num_batches_tracked' in key:
            avg[key] = int(np.max(arrays))   # tomar el máximo
        else:
            avg[key] = np.mean(arrays, axis=0)
    return avg


def _clip_gradients_by_norm(grads: dict, max_norm: float) -> tuple[dict, float]:
    """Gradient clipping por norma global."""
    total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))
    clip_coef  = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        grads = {k: v * clip_coef for k, v in grads.items()}
    return grads, total_norm


def _apply_momentum(weights: dict, grads: dict, velocity: dict,
                    lr: float, momentum: float) -> tuple[dict, dict]:
    """SGD con momentum. Buffers sin gradiente se copian sin cambio."""
    new_weights  = {}
    new_velocity = {}
    for key in weights:
        if key not in grads:
            new_weights[key]  = weights[key]
            new_velocity[key] = velocity.get(key, np.zeros_like(
                np.array(weights[key], dtype=np.float32)))
        else:
            g = np.array(grads[key], dtype=np.float32)
            v = momentum * velocity.get(key, np.zeros_like(g)) + lr * g
            new_weights[key]  = np.array(weights[key], dtype=np.float32) - v
            new_velocity[key] = v
    return new_weights, new_velocity


async def send_weights_to_worker(writer, weights: dict, epoch: int):
    serializable = {
        k: (v.tolist() if isinstance(v, np.ndarray) else
            v.cpu().numpy().tolist() if hasattr(v, 'cpu') else
            int(v) if isinstance(v, (np.integer,)) else v)
        for k, v in weights.items()
    }
    await send_json(writer, {
        "type":       "weights",
        "epoch":      epoch,
        "model_type": state.model_type,
        "weights":    serializable,
    })


# ── Bucle principal ───────────────────────────────────────────────────────────

async def training_loop():
    print("\n" + "=" * 70)
    print("INICIO DE ENTRENAMIENTO")
    print("=" * 70)

    # Ajuste automático de lr
    if state.model_type == 'cnn' and state.learning_rate >= _CNN_LR_MAX:
        old_lr = state.learning_rate
        state.learning_rate = _CNN_LR_DEFAULT
        print(f"⚠  lr={old_lr} ajustado a {_CNN_LR_DEFAULT} para CNN federada")

    print(f"   Optimizador: SGD momentum={_MOMENTUM}, "
          f"lr={state.learning_rate}, grad_clip={_GRAD_NORM_CLIP}")

    async with state.lock:
        state.worker_gradients    = {}
        state.worker_losses       = {}
        state.all_workers_ready   = asyncio.Event()
        state.current_epoch       = 0
        state.training_start_time = time.time()
        state.epoch_events        = {}

    # Pesos globales desde state_dict completo
    if state.model_type == 'cnn' and state.model:
        state.global_weights = {
            k: v.cpu().numpy()
            for k, v in state.model.model.state_dict().items()
        }
        print(f"✓ Pesos CNN desde state_dict "
              f"({len(state.global_weights)} tensores)")

    # Acumulador de momentum solo para tensores float
    velocity = {
        k: np.zeros_like(np.array(v, dtype=np.float32))
        for k, v in state.global_weights.items()
        if 'num_batches_tracked' not in k
    }

    # Esperar workers
    print("Esperando que todos los workers completen su setup...")
    while True:
        async with state.lock:
            if state.check_all_workers_ready_for_training():
                break
        await asyncio.sleep(0.5)
    print("✓ Todos los workers listos\n")

    init_loss, init_acc = _evaluate()
    print(f"Evaluación inicial — Loss:{init_loss:.4f}  Accuracy:{init_acc:.4f}\n")

    # ── Épocas ────────────────────────────────────────────────────────────────
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

        async with state.lock:
            workers_snapshot = list(state.worker_writers.items())

        print(f"\n[Época {epoch+1}] Enviando pesos...")
        for wid, writer in workers_snapshot:
            try:
                await send_weights_to_worker(
                    writer, state.global_weights, epoch + 1)
                print(f"   Pesos → Worker {wid}")
            except Exception as e:
                print(f"   ✗ Error enviando a Worker {wid}: {e}")

        print(f"\n[Época {epoch+1}] Esperando gradientes...")
        try:
            async with state.lock:
                ready_event = state.all_workers_ready
            await asyncio.wait_for(ready_event.wait(), timeout=6000.0)

            async with state.lock:
                gradients_list         = list(state.worker_gradients.values())
                losses_list            = list(state.worker_losses.values())
                worker_losses_snapshot = dict(state.worker_losses)
                # Buffers de BatchNorm enviados por los workers (puede ser vacío
                # si los workers no los envían aún — compatibilidad hacia atrás)
                bn_buffers_list = [
                    g.get('__bn_buffers__', {})
                    for g in state.worker_gradients.values()
                ]

            # Separar gradientes reales de los buffers bn si vienen mezclados
            clean_grads = []
            for g in gradients_list:
                clean_grads.append({k: v for k, v in g.items()
                                    if k != '__bn_buffers__'})

            # Agregar gradientes con clipping
            avg_grads            = _average_gradients(clean_grads)
            avg_grads, grad_norm = _clip_gradients_by_norm(
                avg_grads, _GRAD_NORM_CLIP)

            # Actualizar pesos con momentum
            state.global_weights, velocity = _apply_momentum(
                state.global_weights, avg_grads, velocity,
                lr=state.learning_rate, momentum=_MOMENTUM,
            )

            # Actualizar buffers de BatchNorm con el promedio de los workers
            bn_buffers = _average_bn_buffers(
                [b for b in bn_buffers_list if b])
            if bn_buffers:
                for key, val in bn_buffers.items():
                    state.global_weights[key] = val

            avg_loss            = float(np.mean(losses_list))
            test_loss, test_acc = _evaluate()

            async with state.lock:
                epoch_time = time.time() - state.epoch_start_time

            state.train_losses.append(avg_loss)
            state.test_losses.append(test_loss)
            state.test_accuracies.append(test_acc)
            state.epoch_times.append(epoch_time)

            async with state.lock:
                for wid, lv in worker_losses_snapshot.items():
                    state.worker_loss_history.setdefault(wid, []).append(lv)

            save_epoch_metrics(epoch, avg_loss, test_loss, test_acc,
                               epoch_time, worker_losses_snapshot)

            clip_tag = '  ← clippeado' if grad_norm > _GRAD_NORM_CLIP else ''
            print(f"\n RESULTADOS ÉPOCA {epoch + 1}:")
            print(f"  Train loss:       {avg_loss:.4f}")
            print(f"  Test  loss:       {test_loss:.4f}")
            print(f"  Test  accuracy:   {test_acc:.4f}")
            print(f"  Norma gradientes: {grad_norm:.2f}{clip_tag}")
            print(f"  Tiempo época:     {epoch_time:.2f}s")

        except asyncio.TimeoutError:
            async with state.lock:
                received = len(state.worker_gradients)
                expected = len(state.worker_writers)
                missing  = set(state.worker_writers.keys()) - \
                           set(state.worker_gradients.keys())
            print(f"\n✗ Timeout época {epoch+1}: "
                  f"{received}/{expected} respondieron")
            print(f"  Workers faltantes: {missing}")
            break

        except Exception as e:
            print(f"\n✗ Error en época {epoch+1}: {e}")
            import traceback; traceback.print_exc()
            break

        print()

    # ── Final ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 70)
    final_loss, final_acc = _evaluate()
    total_time = time.time() - state.training_start_time
    print(f"Loss final: {final_loss:.4f} | Accuracy final: {final_acc:.4f}")
    print(f"Tiempo total: {total_time:.2f}s")
    print("=" * 70)

    save_final_summary(final_loss, final_acc, total_time)