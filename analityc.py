# visualize_metrics.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuración de estilo para mejores gráficas
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

def load_and_prepare_data(filepath='training_results/experiments_summary.csv'):
    """Carga y prepara los datos para análisis."""
    df = pd.read_csv(filepath)
    
    # Convertir columnas a tipos numéricos
    df['workers'] = pd.to_numeric(df['workers'])
    df['total_time_minutes'] = pd.to_numeric(df['total_time_minutes'])
    df['final_test_accuracy'] = pd.to_numeric(df['final_test_accuracy'])
    df['learning_rate'] = pd.to_numeric(df['learning_rate'])
    df['epochs'] = pd.to_numeric(df['epochs'])
    
    # Extraer timestamp para ordenar
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
    
    # Calcular métricas adicionales
    df['accuracy_per_hour'] = df['final_test_accuracy'] / (df['total_time_minutes'] / 60)
    df['time_per_epoch_seconds'] = df['total_time_seconds'] / df['epochs']
    
    return df

def plot_dataset_comparison(df):
    """Gráfica específica para comparar datasets."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Comparativa de Rendimiento: MNIST vs CIFAR-10', fontsize=16, fontweight='bold')
    
    # 1. Tiempo total por dataset (barras agrupadas)
    ax1 = axes[0, 0]
    datasets = df['dataset'].unique()
    x = np.arange(len(datasets))
    width = 0.35
    
    # Agrupar por workers
    for workers in sorted(df['workers'].unique()):
        times = []
        for dataset in datasets:
            subset = df[(df['dataset'] == dataset) & (df['workers'] == workers)]
            if not subset.empty:
                times.append(subset['total_time_minutes'].mean())
            else:
                times.append(0)
        
        ax1.bar(x + width * (workers - 1), times, width, label=f'{workers} worker(s)')
    
    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('Tiempo Total (minutos)')
    ax1.set_title('Tiempo de Entrenamiento por Dataset')
    ax1.set_xticks(x)
    ax1.set_xticklabels([d.upper() for d in datasets])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Accuracy por dataset (barras agrupadas)
    ax2 = axes[0, 1]
    for workers in sorted(df['workers'].unique()):
        accuracies = []
        for dataset in datasets:
            subset = df[(df['dataset'] == dataset) & (df['workers'] == workers)]
            if not subset.empty:
                accuracies.append(subset['final_test_accuracy'].mean())
            else:
                accuracies.append(0)
        
        ax2.bar(x + width * (workers - 1), accuracies, width, label=f'{workers} worker(s)')
    
    ax2.set_xlabel('Dataset')
    ax2.set_ylabel('Accuracy Final')
    ax2.set_title('Precisión del Modelo por Dataset')
    ax2.set_xticks(x)
    ax2.set_xticklabels([d.upper() for d in datasets])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 1])
    
    # 3. Tiempo vs Workers (líneas separadas por dataset)
    ax3 = axes[1, 0]
    markers = ['o', 's', '^', 'D']
    for i, dataset in enumerate(datasets):
        subset = df[df['dataset'] == dataset].sort_values('workers')
        ax3.plot(subset['workers'], subset['total_time_minutes'], 
                marker=markers[i % len(markers)], linewidth=2, markersize=8,
                label=dataset.upper())
    
    ax3.set_xlabel('Número de Workers')
    ax3.set_ylabel('Tiempo Total (minutos)')
    ax3.set_title('Escalabilidad: Tiempo vs Workers por Dataset')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Accuracy vs Workers (líneas separadas por dataset)
    ax4 = axes[1, 1]
    for i, dataset in enumerate(datasets):
        subset = df[df['dataset'] == dataset].sort_values('workers')
        ax4.plot(subset['workers'], subset['final_test_accuracy'], 
                marker=markers[i % len(markers)], linewidth=2, markersize=8,
                label=dataset.upper())
    
    ax4.set_xlabel('Número de Workers')
    ax4.set_ylabel('Accuracy Final')
    ax4.set_title('Calidad del Modelo vs Workers por Dataset')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('training_results/dataset_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_comparative_analysis(df):
    """Genera gráficas comparativas para analizar el rendimiento."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Análisis de Experimentos', fontsize=16, fontweight='bold')
    
    # 1. Tiempo vs Workers
    ax1 = axes[0, 0]
    for dataset in df['dataset'].unique():
        subset = df[df['dataset'] == dataset]
        ax1.plot(subset['workers'], subset['total_time_minutes'], 'o-', 
                linewidth=2, markersize=8, label=dataset.upper())
    ax1.set_xlabel('Número de Workers')
    ax1.set_ylabel('Tiempo Total (minutos)')
    ax1.set_title('Escalabilidad: Tiempo vs Workers')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Accuracy vs Workers
    ax2 = axes[0, 1]
    for dataset in df['dataset'].unique():
        subset = df[df['dataset'] == dataset]
        ax2.plot(subset['workers'], subset['final_test_accuracy'], 's-', 
                linewidth=2, markersize=8, label=dataset.upper())
    ax2.set_xlabel('Número de Workers')
    ax2.set_ylabel('Accuracy Final')
    ax2.set_title('Calidad del Modelo vs Workers')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # 3. Accuracy vs Tiempo (trade-off)
    ax3 = axes[1, 0]
    colors = plt.cm.Set1(np.linspace(0, 1, len(df['workers'].unique())))
    for i, workers in enumerate(sorted(df['workers'].unique())):
        subset = df[df['workers'] == workers]
        ax3.scatter(subset['total_time_minutes'], subset['final_test_accuracy'], 
                   s=100, color=colors[i], label=f'{workers} worker(s)', alpha=0.7)
        
        # Añadir etiquetas con dataset y timestamp
        for _, row in subset.iterrows():
            time_str = row['timestamp'].split()[1][:5]  # HH:MM
            label = f"{row['dataset'][:3].upper()}-{time_str}"
            ax3.annotate(label, 
                        (row['total_time_minutes'], row['final_test_accuracy']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax3.set_xlabel('Tiempo Total (minutos)')
    ax3.set_ylabel('Accuracy Final')
    ax3.set_title('Trade-off: Accuracy vs Tiempo de Entrenamiento')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Evolución temporal de experiments
    ax4 = axes[1, 1]
    df_sorted = df.sort_values('timestamp_dt')
    for dataset in df_sorted['dataset'].unique():
        subset = df_sorted[df_sorted['dataset'] == dataset]
        ax4.plot(range(len(subset)), subset['final_test_accuracy'], 'o-', 
                linewidth=2, markersize=8, label=dataset.upper())
    
    ax4.set_xlabel('Orden de Ejecución')
    ax4.set_ylabel('Accuracy Final')
    ax4.set_title('Evolución del Rendimiento por Experimento')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_results/comparative_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_statistical_analysis(df):
    """Análisis estadístico de los experimentos."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Análisis Estadístico de Experimentos', fontsize=14, fontweight='bold')
    
    # Boxplot de accuracy por workers y dataset
    ax1 = axes[0, 0]
    df.boxplot(column='final_test_accuracy', by=['workers', 'dataset'], ax=ax1, rot=45)
    ax1.set_title('Distribución de Accuracy por Workers y Dataset')
    ax1.set_xlabel('Configuración')
    ax1.set_ylabel('Accuracy Final')
    ax1.grid(True, alpha=0.3)
    
    # Boxplot de tiempo por workers y dataset
    ax2 = axes[0, 1]
    df.boxplot(column='total_time_minutes', by=['workers', 'dataset'], ax=ax2, rot=45)
    ax2.set_title('Distribución de Tiempo por Workers y Dataset')
    ax2.set_xlabel('Configuración')
    ax2.set_ylabel('Tiempo Total (minutos)')
    ax2.grid(True, alpha=0.3)
    
    # Eficiencia: Accuracy por hora
    ax3 = axes[1, 0]
    for dataset in df['dataset'].unique():
        subset = df[df['dataset'] == dataset]
        ax3.bar(subset['workers'], subset['accuracy_per_hour'], 
                alpha=0.7, label=dataset.upper(), width=0.35)
    ax3.set_xlabel('Número de Workers')
    ax3.set_ylabel('Accuracy por Hora')
    ax3.set_title('Eficiencia: Accuracy ganada por hora de entrenamiento')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Tiempo por época
    ax4 = axes[1, 1]
    for dataset in df['dataset'].unique():
        subset = df[df['dataset'] == dataset]
        ax4.plot(subset['workers'], subset['time_per_epoch_seconds'], 'o-', 
                linewidth=2, markersize=8, label=dataset.upper())
    ax4.set_xlabel('Número de Workers')
    ax4.set_ylabel('Tiempo por Época (segundos)')
    ax4.set_title('Rendimiento: Tiempo por Época')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_results/statistical_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

def print_summary_statistics(df):
    """Imprime estadísticas resumidas de los experimentos."""
    
    print("\n" + "="*70)
    print("RESUMEN DE EXPERIMENTOS")
    print("="*70)
    
    print(f"\n Total de experimentos: {len(df)}")
    print(f" Datasets: {', '.join(df['dataset'].unique().upper())}")
    
    # Estadísticas por dataset
    print("\n ESTADÍSTICAS POR DATASET:")
    print("-"*70)
    
    for dataset in df['dataset'].unique():
        subset = df[df['dataset'] == dataset]
        print(f"\n🔹 {dataset.upper()}:")
        print(f"   - Experimentos: {len(subset)}")
        print(f"   - Mejor Accuracy: {subset['final_test_accuracy'].max():.4f}")
        print(f"   - Peor Accuracy: {subset['final_test_accuracy'].min():.4f}")
        print(f"   - Accuracy promedio: {subset['final_test_accuracy'].mean():.4f} ± {subset['final_test_accuracy'].std():.4f}")
        print(f"   - Tiempo promedio: {subset['total_time_minutes'].mean():.1f} min ± {subset['total_time_minutes'].std():.1f}")
        print(f"   - Tiempo por época: {subset['time_per_epoch_seconds'].mean():.1f} seg")
    
    print("\n ESTADÍSTICAS POR NÚMERO DE WORKERS:")
    print("-"*70)
    
    stats = df.groupby('workers').agg({
        'final_test_accuracy': ['mean', 'std', 'min', 'max'],
        'total_time_minutes': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    print(stats)
    
    print("\n MEJORES RESULTADOS:")
    print("-"*70)
    
    # Mejor accuracy
    best_acc = df.loc[df['final_test_accuracy'].idxmax()]
    print(f" Mejor Accuracy: {best_acc['final_test_accuracy']:.4f}")
    print(f"   - Dataset: {best_acc['dataset'].upper()}")
    print(f"   - Workers: {best_acc['workers']}")
    print(f"   - Épocas: {best_acc['epochs']}")
    print(f"   - Tiempo: {best_acc['total_time_minutes']:.2f} min")
    print(f"   - Learning Rate: {best_acc['learning_rate']}")
    
    # Mejor tiempo
    best_time = df.loc[df['total_time_minutes'].idxmin()]
    print(f"\n Mejor Tiempo: {best_time['total_time_minutes']:.2f} minutos")
    print(f"   - Dataset: {best_time['dataset'].upper()}")
    print(f"   - Accuracy: {best_time['final_test_accuracy']:.4f}")
    print(f"   - Workers: {best_time['workers']}")
    
    # Mayor eficiencia (accuracy por hora)
    best_efficiency = df.loc[df['accuracy_per_hour'].idxmax()]
    print(f"\n Mayor Eficiencia: {best_efficiency['accuracy_per_hour']:.4f} accuracy/hora")
    print(f"   - Dataset: {best_efficiency['dataset'].upper()}")
    print(f"   - Accuracy: {best_efficiency['final_test_accuracy']:.4f}")
    print(f"   - Tiempo: {best_efficiency['total_time_minutes']:.2f} min")
    print(f"   - Workers: {best_efficiency['workers']}")
    
    print("\n" + "="*70)

def plot_learning_rate_comparison(df):
    """Compara el efecto del learning rate si hay diferentes valores."""
    
    if len(df['learning_rate'].unique()) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Impacto del Learning Rate', fontsize=14, fontweight='bold')
        
        # Accuracy vs Learning Rate
        ax1 = axes[0]
        for dataset in df['dataset'].unique():
            subset = df[df['dataset'] == dataset]
            ax1.scatter(subset['learning_rate'], subset['final_test_accuracy'], 
                       s=100, label=dataset.upper(), alpha=0.7)
        ax1.set_xlabel('Learning Rate')
        ax1.set_ylabel('Accuracy Final')
        ax1.set_title('Accuracy vs Learning Rate')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        
        # Tiempo vs Learning Rate
        ax2 = axes[1]
        for dataset in df['dataset'].unique():
            subset = df[df['dataset'] == dataset]
            ax2.scatter(subset['learning_rate'], subset['total_time_minutes'], 
                       s=100, label=dataset.upper(), alpha=0.7)
        ax2.set_xlabel('Learning Rate')
        ax2.set_ylabel('Tiempo Total (minutos)')
        ax2.set_title('Tiempo vs Learning Rate')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig('training_results/learning_rate_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
    else:
        print(f"\nℹ Todos los experimentos usaron el mismo learning rate: {df['learning_rate'].iloc[0]}")

def main():
    """Función principal de visualización."""
    
    print(" Generando visualizaciones de experimentos...")
    
    # Cargar datos
    try:
        df = load_and_prepare_data()
    except FileNotFoundError:
        print(" No se encontró el archivo training_results/experiments_summary.csv")
        print("   Ejecuta primero algunos entrenamientos para generar datos.")
        return
    
    # Mostrar datos
    print("\n Datos cargados:")
    print(df[['timestamp', 'dataset', 'workers', 'epochs', 'final_test_accuracy', 'total_time_minutes']].to_string())
    
    # Análisis estadístico
    print_summary_statistics(df)
    
    # Generar gráficas
    plot_dataset_comparison(df)      # NUEVA: Comparativa específica de datasets
    plot_comparative_analysis(df)     # Análisis general
    plot_statistical_analysis(df)     # Análisis estadístico mejorado
    plot_learning_rate_comparison(df) # Comparación de learning rate
    
    print("\n Análisis completado!")
    print(" Gráficas guardadas en: training_results/")
    print("   - dataset_comparison.png (Comparativa MNIST vs CIFAR-10)")
    print("   - comparative_analysis.png (Análisis general)")
    print("   - statistical_analysis.png (Análisis estadístico)")
    print("   - learning_rate_comparison.png (Impacto del LR)")

if __name__ == "__main__":
    main()