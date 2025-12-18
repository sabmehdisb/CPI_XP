import argparse
import glob
import json
import os
import sys

def format_float(value, precision=1):
    if value is None:
        return "N/A"
    return f"{value:.{precision}f}"

def format_time(value):
    """Formatte le temps (scientifique si < 0.001, sinon 3 décimales)."""
    if value is None:
        return "-"
    if value == 0:
        return "0"
    if value < 0.001:
        return f"{value:.1e}"
    return f"{value:.3f}"

def load_json_files(directory):
    json_files = glob.glob(os.path.join(directory, "*.json"))
    data_list = []
    for file_path in sorted(json_files):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data_list.append(data)
        except Exception as e:
            print(f"Erreur lecture {file_path}: {e}", file=sys.stderr)
    return data_list

def generate_dataset_table(data_list):
    """Génère le Tableau 1 (Style booktabs, Caption en bas)."""
    latex = []
    
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\small")
    latex.append(r"\setlength{\tabcolsep}{4pt}")
    
    # Resizebox pour adapter à la colonne
    latex.append(r"\resizebox{\columnwidth}{!}{")
    
    latex.append(r"\begin{tabular}{lrrr}")
    latex.append(r"\toprule")
    latex.append(r"\textbf{Dataset} & \textbf{$|D|$} & \textbf{$|A|$} & \textbf{$X_{DT}$} \\")
    latex.append(r"\midrule")
    
    for entry in data_list:
        name = entry.get("dataset", "Unk").replace("_", r"\_")
        # Raccourcir les noms trop longs SI NÉCESSAIRE (optionnel)
        if len(name) > 25: 
             name = name[:23] + ".."

        meta = entry.get("dataset_metadata", {})
        n_instances = meta.get("n_instances", meta.get("total_rows", "-"))
        n_features = meta.get("n_features", "-")
        avg_binary = meta.get("avg_binary_features_size", 0)
        
        latex.append(f"{name} & {n_instances} & {n_features} & {format_float(avg_binary, 1)} \\\\")
        
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"}") # Fin resizebox
    
    # Caption EN BAS
    latex.append(r"\caption{Dataset statistics. $|D|$: Instances, $|A|$: Features, $X_{DT}$: Avg. binary features.}")
    latex.append(r"\label{tab:datasets}")
    
    latex.append(r"\end{table}")
    
    return "\n".join(latex)

def generate_comparison_table(data_list):
    """Génère le Tableau 2 (Style booktabs, Caption en bas, Gras si TO > 0)."""
    latex = []
    
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\small")
    latex.append(r"\setlength{\tabcolsep}{2.5pt}")
    
    # Resizebox pour adapter à la colonne
    latex.append(r"\resizebox{\columnwidth}{!}{")
    
    latex.append(r"\begin{tabular}{lrrrrrr}")
    latex.append(r"\toprule")
    
    # En-têtes groupés avec cmidrule
    latex.append(r"& \multicolumn{3}{c}{\textbf{Cooper et al.}} & \multicolumn{3}{c}{\textbf{PyXAI (CPI-XP)}} \\")
    latex.append(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}")
    
    latex.append(r"\textbf{Dataset} & \textbf{Time} & \textbf{TO\%} & \textbf{Sz} & \textbf{Time} & \textbf{TO\%} & \textbf{Sz} \\")
    latex.append(r"\midrule")
    
    for entry in data_list:
        name = entry.get("dataset", "Unk").replace("_", r"\_")
        
        # --- CORRECTION ICI ---
        # J'ai retiré .replace("balance\_", "")
        # Je garde juste le raccourcissement de compas si vous le souhaitez, sinon enlevez-le aussi.
        name_clean = name.replace("compas", "comp")
        
        methods = entry.get("methods", {})
        ext = methods.get("external_cooper_amgoud", {})
        cpi = methods.get("pyxai_cpi_xp", {})
        
        # --- Valeurs ---
        ext_t = ext.get("mean_time_s")
        ext_to_val = ext.get("timeout_percentage", 0)
        ext_sz = ext.get("mean_feature_size")
        
        cpi_t = cpi.get("mean_time_s")
        cpi_to_val = cpi.get("timeout_percentage", 0)
        cpi_sz = cpi.get("mean_feature_size")
        
        # --- Formatage Temps (Gras pour le meilleur) ---
        t_ext_str = format_time(ext_t)
        t_cpi_str = format_time(cpi_t)
        
        if cpi_t is not None and ext_t is not None:
            if cpi_t < ext_t:
                t_ext_str = r"\textbf{" + t_ext_str + "}"

        # --- Formatage Timeout (Gras si > 0) ---
        # External
        to_ext_str = f"{int(ext_to_val)}" if ext_to_val == int(ext_to_val) else f"{ext_to_val:.1f}"
        if ext_to_val > 0:
            to_ext_str = r"\textbf{" + to_ext_str + "}"
            
        # PyXAI
        to_cpi_str = f"{int(cpi_to_val)}" if cpi_to_val == int(cpi_to_val) else f"{cpi_to_val:.1f}"
        if cpi_to_val > 0:
            to_cpi_str = r"\textbf{" + to_cpi_str + "}"

        # --- Formatage Taille ---
        sz_ext_str = format_float(ext_sz, 1)
        sz_cpi_str = format_float(cpi_sz, 1)
        
        # Ligne du tableau
        row = (f"{name_clean} & "
               f"{t_ext_str} & {to_ext_str} & {sz_ext_str} & "
               f"{t_cpi_str} & {to_cpi_str} & {sz_cpi_str} \\\\")
        latex.append(row)
        
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"}") # Fin resizebox
    
    # Caption EN BAS
    latex.append(r"\caption{Comparison: Cooper et al.(CPI-XP) vs PyXAI (CPI-XP). Time in sec. Timeouts (TO) $>0$\% are in bold.}")
    latex.append(r"\label{tab:comparison}")
    
    latex.append(r"\end{table}")
    
    return "\n".join(latex)

def main():
    parser = argparse.ArgumentParser(description="Générateur LaTeX IJCAI Booktabs")
    parser.add_argument("--path", required=True, help="Dossier des logs JSON")
    args = parser.parse_args()
    
    if not os.path.exists(args.path):
        print("Dossier introuvable.")
        return

    data = load_json_files(args.path)
    if not data:
        print("Aucun JSON valide trouvé.")
        return

    print("% ================= TABLE 1 =================\n")
    print(generate_dataset_table(data))
    print("\n\n")
    print("% ================= TABLE 2 =================\n")
    print(generate_comparison_table(data))

if __name__ == "__main__":
    main()