# Export_Metrics.py - Servidor
import csv
import os
from datetime import datetime
from Config import state

# El timestamp se fija al importar el módulo para que todas las épocas
# del mismo experimento escriban en el MISMO archivo.
_EXPERIMENT_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
_DETAILED_FILE = None   # se asigna en la primera llamada a save_epoch_metrics


def _get_detailed_file() -> str:
    global _DETAILED_FILE
    if _DETAILED_FILE is None:
        results_dir = "training_results"
        os.makedirs(results_dir, exist_ok=True)
        _DETAILED_FILE = (
            f"{results_dir}/detailed_"
            f"{state.dataset_name}_{state.expected_workers}w_"
            f"{state.max_epochs}e_{_EXPERIMENT_TIMESTAMP}.csv"
        )
    return _DETAILED_FILE


def save_epoch_metrics(epoch: int, avg_loss: float, test_loss: float,
                       test_acc: float, epoch_time: float,
                       worker_losses: dict) -> str | None:
    """Guarda las métricas de cada época en el archivo CSV del experimento."""
    detailed_file = _get_detailed_file()
    is_first = (epoch == 0)

    try:
        with open(detailed_file, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)

            if is_first:
                w.writerow(["=== METADATOS ==="])
                w.writerow(["timestamp",    _EXPERIMENT_TIMESTAMP])
                w.writerow(["dataset",      state.dataset_name])
                w.writerow(["model_type",   state.model_type])
                w.writerow(["input_size",   state.input_size])
                w.writerow(["workers",      state.expected_workers])
                w.writerow(["epochs",       state.max_epochs])
                w.writerow(["learning_rate",state.learning_rate])
                w.writerow([])
                w.writerow(["=== MÉTRICAS POR ÉPOCA ==="])
                header = ["epoch", "train_loss", "test_loss",
                          "test_accuracy", "epoch_time_s"]
                if worker_losses:
                    header += [f"worker_{wid}_loss"
                               for wid in sorted(worker_losses)]
                w.writerow(header)

            row = [epoch + 1, f"{avg_loss:.6f}", f"{test_loss:.6f}",
                   f"{test_acc:.6f}", f"{epoch_time:.3f}"]
            if worker_losses:
                for wid in sorted(worker_losses):
                    row.append(f"{worker_losses[wid]:.6f}")
            w.writerow(row)

        return detailed_file

    except Exception as e:
        print(f"✗ Error guardando métricas de época: {e}")
        return None


def save_final_summary(final_loss: float, final_acc: float,
                       total_time: float) -> str | None:
    """Agrega una fila al CSV acumulativo de experimentos."""
    results_dir = "training_results"
    os.makedirs(results_dir, exist_ok=True)
    summary_file = f"{results_dir}/experiments_summary.csv"
    file_exists  = os.path.isfile(summary_file)

    data = {
        'timestamp':          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'dataset':            state.dataset_name,
        'model_type':         state.model_type,
        'workers':            state.expected_workers,
        'epochs':             state.max_epochs,
        'learning_rate':      state.learning_rate,
        'input_size':         state.input_size,
        'total_time_seconds': f"{total_time:.3f}",
        'total_time_minutes': f"{total_time / 60:.2f}",
        'final_test_loss':    f"{final_loss:.6f}",
        'final_test_accuracy':f"{final_acc:.6f}",
    }

    try:
        with open(summary_file, 'a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=data.keys())
            if not file_exists:
                w.writeheader()
                print(f"✓ Creando archivo de resumen: {summary_file}")
            w.writerow(data)
        print(f"✓ Resumen agregado a: {summary_file}")
        return summary_file
    except Exception as e:
        print(f"✗ Error exportando resumen: {e}")
        return None