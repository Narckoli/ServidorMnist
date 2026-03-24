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
    
    # Extraer timestamp para ordenar
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
    
    return df

def plot_comparative_analysis(df):
    """Genera gráficas comparativas para analizar el rendimiento."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Análisis de Experimentos - CIFAR-10', fontsize=16, fontweight='bold')
    
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
        
        # Añadir etiquetas de timestamp para cada punto
        for _, row in subset.iterrows():
            time_str = row['timestamp'].split()[1][:5]  # HH:MM
            ax3.annotate(time_str, 
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
    for workers in df_sorted['workers'].unique():
        subset = df_sorted[df_sorted['workers'] == workers]
        ax4.plot(range(len(subset)), subset['final_test_accuracy'], 'o-', 
                linewidth=2, markersize=8, label=f'{workers} worker(s)')
    
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
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Análisis Estadístico de Experimentos', fontsize=14, fontweight='bold')
    
    # Boxplot de accuracy por workers
    ax1 = axes[0]
    df.boxplot(column='final_test_accuracy', by='workers', ax=ax1)
    ax1.set_title('Distribución de Accuracy por Número de Workers')
    ax1.set_xlabel('Número de Workers')
    ax1.set_ylabel('Accuracy Final')
    ax1.grid(True, alpha=0.3)
    
    # Boxplot de tiempo por workers
    ax2 = axes[1]
    df.boxplot(column='total_time_minutes', by='workers', ax=ax2)
    ax2.set_title('Distribución de Tiempo por Número de Workers')
    ax2.set_xlabel('Número de Workers')
    ax2.set_ylabel('Tiempo Total (minutos)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_results/statistical_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

def print_summary_statistics(df):
    """Imprime estadísticas resumidas de los experimentos."""
    
    print("\n" + "="*70)
    print("RESUMEN DE EXPERIMENTOS")
    print("="*70)
    
    print(f"\n📊 Total de experimentos: {len(df)}")
    print(f"📁 Dataset: {df['dataset'].unique()[0].upper()}")
    
    print("\n📈 ESTADÍSTICAS POR NÚMERO DE WORKERS:")
    print("-"*70)
    
    stats = df.groupby('workers').agg({
        'final_test_accuracy': ['mean', 'std', 'min', 'max'],
        'total_time_minutes': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    print(stats)
    
    print("\n⏱️ MEJORES RESULTADOS:")
    print("-"*70)
    
    # Mejor accuracy
    best_acc = df.loc[df['final_test_accuracy'].idxmax()]
    print(f"🎯 Mejor Accuracy: {best_acc['final_test_accuracy']:.4f}")
    print(f"   - Workers: {best_acc['workers']}")
    print(f"   - Tiempo: {best_acc['total_time_minutes']:.2f} min")
    print(f"   - Épocas: {best_acc['epochs']}")
    print(f"   - Learning Rate: {best_acc['learning_rate']}")
    print(f"   - Timestamp: {best_acc['timestamp']}")
    
    # Mejor tiempo
    best_time = df.loc[df['total_time_minutes'].idxmin()]
    print(f"\n⚡ Mejor Tiempo: {best_time['total_time_minutes']:.2f} minutos")
    print(f"   - Accuracy: {best_time['final_test_accuracy']:.4f}")
    print(f"   - Workers: {best_time['workers']}")
    
    # Trade-off (mejor accuracy por minuto)
    df['accuracy_per_minute'] = df['final_test_accuracy'] / df['total_time_minutes']
    best_tradeoff = df.loc[df['accuracy_per_minute'].idxmax()]
    print(f"\n⚖️ Mejor Trade-off (Accuracy/minuto): {best_tradeoff['accuracy_per_minute']:.4f}")
    print(f"   - Accuracy: {best_tradeoff['final_test_accuracy']:.4f}")
    print(f"   - Tiempo: {best_tradeoff['total_time_minutes']:.2f} min")
    print(f"   - Workers: {best_tradeoff['workers']}")
    
    print("\n" + "="*70)

def plot_learning_rate_comparison(df):
    """Compara el efecto del learning rate si hay diferentes valores."""
    
    if len(df['learning_rate'].unique()) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for lr in df['learning_rate'].unique():
            subset = df[df['learning_rate'] == lr]
            ax.plot(subset['workers'], subset['final_test_accuracy'], 'o-', 
                   linewidth=2, markersize=8, label=f'LR = {lr}')
        
        ax.set_xlabel('Número de Workers')
        ax.set_ylabel('Accuracy Final')
        ax.set_title('Impacto del Learning Rate en el Rendimiento')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_results/learning_rate_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
    else:
        print(f"\nℹ️ Todos los experimentos usaron el mismo learning rate: {df['learning_rate'].iloc[0]}")

def main():
    """Función principal de visualización."""
    
    print("📊 Generando visualizaciones de experimentos...")
    
    # Cargar datos
    try:
        df = load_and_prepare_data()
    except FileNotFoundError:
        print("❌ No se encontró el archivo training_results/experiments_summary.csv")
        print("   Ejecuta primero algunos entrenamientos para generar datos.")
        return
    
    # Mostrar datos
    print("\n📋 Datos cargados:")
    print(df[['timestamp', 'workers', 'final_test_accuracy', 'total_time_minutes']].to_string())
    
    # Análisis estadístico
    print_summary_statistics(df)
    
    # Generar gráficas
    plot_comparative_analysis(df)
    plot_statistical_analysis(df)
    plot_learning_rate_comparison(df)
    
    print("\n✅ Análisis completado!")
    print("📁 Gráficas guardadas en: training_results/")

if __name__ == "__main__":
    main()