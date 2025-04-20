import os
import pandas as pd
import csv


def import_data(input_directory: str):
    # Lecture des fichiers CSV
    nodes_df = pd.read_csv(os.path.join(input_directory, "nodes.csv"))
    tasks_df = pd.read_csv(os.path.join(input_directory, "tasks.csv"))
    subtasks_df = pd.read_csv(os.path.join(input_directory, "subtasks.csv"))
    arcs_df = pd.read_csv(os.path.join(input_directory, "arcs.csv"))
    robots_df = pd.read_csv(os.path.join(input_directory, "robots.csv"))
    refueling_arc_df = pd.read_csv(os.path.join(input_directory, "refueling_arcs.csv"))
    f_dij_df = pd.read_csv(os.path.join(input_directory, "f_dij.csv"))

    # Convertir les DataFrames en dictionnaires
    N = {row["Node"]: row["ID"] for _, row in nodes_df.iterrows()}
    B = {row["Task"]: row["Weight"] for _, row in tasks_df.iterrows()}

    B_v = {}
    for _, row in subtasks_df.iterrows():
        task = row["Task"]
        if task not in B_v:
            B_v[task] = {}
        B_v[task][row["SubTask"]] = (row["Node"], row["Time"])

    A = {}
    for _, row in arcs_df.iterrows():
        from_to = f'{row["FromNode"]} => {row["ToNode"]}'
        A[from_to] = (row["Tau"], row["Phi"], row["Psi"])

    A = {k: v for k, v in A.items() if not k.endswith("=> E")}

    D = {
        row["RobotID"]: {"s_d": row["StartingNode"], "F_d": row["MaxFuel"]}
        for _, row in robots_df.iterrows()
    }

    # Extraction de la liste des temps (T)
    T = sorted(subtasks_df["Time"].unique())  # on peut trier pour plus de cohérence

    # Arcs de ravitaillement au cours du temps
    A_Rt = {}
    for _, row in refueling_arc_df.iterrows():
        t = row["Time"]
        arcs = row["Arcs"].split(", ")
        A_Rt[t] = tuple(arcs)

    # f_dij : consommation en carburant pour chaque robot sur chaque arc
    f_dij = {}
    for _, row in f_dij_df.iterrows():
        robot = row["Robot"]
        arc = row["Arc"]
        value = row["Value"]
        if robot not in f_dij:
            f_dij[robot] = {}
        f_dij[robot][arc] = value

    
    # ---------- Lecture de maintenance_windows.csv ----------
    maint_params = {}     # H, h, coûts
    a0 = {}               # operating times par tâche

    mw_path = os.path.join(input_directory, "maintenance_parameters.csv")
    with open(mw_path, newline="") as fp:
        reader = csv.reader(fp)

        # Section 1 : Paramètre / Value
        header = next(reader, None)
        if header != ["Parameter", "Value"]:
            raise ValueError("Entête inattendu dans maintenance_parameters.csv")

        for row in reader:
            if not row:                         # ligne vide → fin de la section 1
                break
            key_txt, val_txt = row[0], row[1]
            key = key_txt.split()[0]            # « H », « h », « c_PM »…
            val = float(val_txt)
            # cast en int si c'est un entier
            maint_params[key] = int(val) if val.is_integer() else val

        # Section 2 : Task / OperatingTime_a0
        header = next(reader, None)           # ['Task', 'OperatingTime_a0']
        for row in reader:
            if not row:
                continue
            task_id   = row[0]                # ex. 'v3'
            a0_value  = int(float(row[1]))    # sûr même si stocké comme 60.0
            a0[task_id] = a0_value

    return (N, B, B_v, A, D, T, A_Rt, f_dij, maint_params, a0)