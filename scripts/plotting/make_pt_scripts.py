import os

# Resolve paths relative to this script's directory
_script_dir = os.path.dirname(os.path.abspath(__file__))

files_to_modify = {
    "plot_metrics.py": [
        ('output_dir = "/home/pedro.vidal/facerec_flower/face_rec_fl/analysis_plots"', 'output_dir = "/home/pedro.vidal/facerec_flower/face_rec_fl/analysis_plots_pt"'),
        ('f"Global Model Accuracy on {ds_name}"', 'f"Acurácia do Modelo Global em {ds_name}"'),
        ('"Communication Round"', '"Rodada de Comunicação"'),
        ('"Accuracy"', '"Acurácia"'),
        ('f"{run} ({len(y_data)} clients)"', 'f"{run} ({len(y_data)} clientes)"'),
        ('"Sorted Final Client Classification Accuracies"', '"Acurácias Finais de Classificação dos Clientes Ordenadas"'),
        ('"Client Percentile (Sorted from Best to Worst)"', '"Percentil de Cliente (Ordenado do Melhor para o Pior)"'),
        ('"Final Local Classification Accuracy"', '"Acurácia Final de Classificação Local"'),
    ],
    "plot_global_loss.py": [
        ('output_dir = "/home/pedro.vidal/facerec_flower/face_rec_fl/analysis_plots"', 'output_dir = "/home/pedro.vidal/facerec_flower/face_rec_fl/analysis_plots_pt"'),
        ('f"{run} Training Loss"', 'f"Perda de Treinamento {run}"'),
        ('"Global Training Loss"', '"Perda de Treinamento Global"'),
        ('"Communication Round"', '"Rodada de Comunicação"'),
        ('"Average Training Loss"', '"Perda de Treinamento Média"'),
        ('f"{run} Training Accuracy"', 'f"Acurácia de Treinamento {run}"'),
        ('"Global Training Accuracy"', '"Acurácia de Treinamento Global"'),
        ('"Average Training Accuracy"', '"Acurácia de Treinamento Média"'),
    ],
    "visualize_drift.py": [
        ('os.makedirs("analysis_plots"', 'os.makedirs("analysis_plots_pt"'),
        ('f"analysis_plots/{run_name}_', 'f"analysis_plots_pt/{run_name}_'),
        ('f"Global Model Weight Divergence (Client Drift) - {run_name}"', 'f"Divergência de Peso do Modelo Global (Desvio do Cliente) - {run_name}"'),
        ('"Communication Round"', '"Rodada de Comunicação"'),
        ('"L2 Norm of Global Update $||w^{(t)} - w^{(t-1)}||_2$"', '"Norma L2 da Atualização Global $||w^{(t)} - w^{(t-1)}||_2$"'),
    ],
    "visualize_global.py": [
        ('os.makedirs("analysis_plots"', 'os.makedirs("analysis_plots_pt"'),
        ('f"analysis_plots/{run_name}_', 'f"analysis_plots_pt/{run_name}_'),
        ("f'{run_name} Path'", "f'Caminho {run_name}'"),
        ("f'Start (Round {valid_rounds[0]})'", "f'Início (Rodada {valid_rounds[0]})'"),
        ("f'End (Round {valid_rounds[-1]})'", "f'Fim (Rodada {valid_rounds[-1]})'"),
        ('f"PCA Global Trajectory - {run_name}"', 'f"Trajetória Global PCA - {run_name}"'),
        ('"Principal Component 1"', '"Componente Principal 1"'),
        ('"Principal Component 2"', '"Componente Principal 2"'),
    ],
    "visualize_layer_updates.py": [
        ('os.makedirs("analysis_plots"', 'os.makedirs("analysis_plots_pt"'),
        ('f"analysis_plots/{run_name}_', 'f"analysis_plots_pt/{run_name}_'),
        ('f"Layer-wise Parameter Adaptation - {run_name} (Round {start_round} $\\\\rightarrow$ {end_round})"', 'f"Adaptação de Parâmetros por Camada - {run_name} (Rodada {start_round} $\\\\rightarrow$ {end_round})"'),
        ('"Relative Change Norm $||\\\\Delta W|| / ||W||$"', '"Norma de Mudança Relativa $||\\\\Delta W|| / ||W||$"'),
    ]
}

for filename, replacements in files_to_modify.items():
    filepath = os.path.join(_script_dir, filename)
    with open(filepath, 'r') as f:
        content = f.read()
    for old, new in replacements:
        content = content.replace(old, new)
    outpath = os.path.join(_script_dir, filename.replace('.py', '_pt.py'))
    with open(outpath, 'w') as f:
        f.write(content)
        print(f"Generated {filename.replace('.py', '_pt.py')}")
