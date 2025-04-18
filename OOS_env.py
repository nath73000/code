import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import os
import pandas as pd
import csv

import gymnasium as gym
from gymnasium import spaces


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
            if not row:           # ligne vide → fin de la section 1
                break
            key_txt, val_txt = row[0], row[1]
            key = key_txt.split()[0]          # « H », « h », « c_PM »…
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




class OOSenv(gym.Env):

    def __init__(self, input_directory: str):
        super(OOSenv, self).__init__()

        # Data importation
        (
            self.N,
            self.B,
            self.B_v,
            self.A,
            self.D,
            self.T,
            self.A_Rt,
            self.f_dij,
            self.maint_params,
            self.a0,
        ) = import_data(input_directory)

        self.num_nodes = len(self.N)

        self.satellites = sorted(list(self.B_v.keys()))
        self.num_satellites = len(self.satellites)

        self.starting_node_str = self.D["d1"]["s_d"]
        self.starting_node = int(self.starting_node_str.replace("n", ""))

        self.max_time_horizon = self.T[-1]

        self.initial_fuel = self.D["d1"]["F_d"]

        self.pos_matrix = np.zeros((self.num_satellites, self.max_time_horizon + 1))
        # We fill the matrix with the position of the satellite for each timestep
        for i, sat_id in enumerate(self.satellites):
            for _, (node, time) in self.B_v[sat_id].items():
                self.pos_matrix[i][time] = int(node.replace("n", ""))

        # On créer une matrice de coût qui correspond aux coûts en step time (:,:,0) et aux coûts en delta V (:,:,1)
        self.cost_matrix = np.zeros((self.num_nodes, self.num_nodes, 2))
        for arc_str, values in self.A.items():
            try:
                parts = arc_str.split(" => ")
                from_node_str = parts[0]
                to_node_str = parts[1]
                from_index = int(from_node_str.replace("n", ""))
                to_index = int(to_node_str.replace("n", ""))
                timecost = float(values[0])
                fuelcost = float(values[1])
                self.cost_matrix[from_index, to_index, 0] = timecost
                self.cost_matrix[from_index, to_index, 1] = fuelcost
            except Exception as e:
                print("Error for arc", arc_str, ":", e)


        # Action sapce :
        self.action_space = gym.spaces.Discrete(self.num_nodes)  # Discrete number of actions from 0 to num_nodes - 1


        # Observation space :
        self.observation_space = gym.spaces.Dict({
            "current_node": gym.spaces.Discrete(self.num_nodes),  # L'agent se trouve sur l'un des noeuds de l'environnement (de 0 a num_nodes - 1)
            "current_fuel": gym.spaces.Box(low=0, high=self.D["d1"]["F_d"], shape=(1,), dtype=np.float32),
            "current_time": gym.spaces.Box(low=0, high=self.max_time_horizon, shape=(1,), dtype=np.int32),
            "visited_satellites": gym.spaces.MultiBinary(self.num_satellites),
            "action_mask": gym.spaces.MultiBinary(self.num_nodes),
            "satellites_pos": gym.spaces.Box(low=0, high=max(self.max_time_horizon, self.num_nodes), shape=(2 * self.num_satellites,), dtype=np.int32,),
        })


    def reset(self, seed=None, options=None):
        self.current_node = self.starting_node
        self.current_time = 0
        self.current_fuel = self.initial_fuel
        self.visited_satellites = np.zeros(self.num_satellites, dtype=np.int8)
        self.route = [self.current_node]
        self.satellites_pos = self.get_satellites_pos()
        return self._get_obs(), {}


    def get_satellites_pos(self):
        pos = []
        for i in range(self.num_satellites):
            pos.append(self.current_time)
            pos.append(int(self.pos_matrix[i][min(self.max_time_horizon, int(self.current_time))]))
        return pos


    def get_allowed_actions(self):
        current_node_str = f"n{self.current_node}"
        allowed = []
        for arc in self.A.keys():
            part = arc.split(" => ")
            if (part[0] == current_node_str 
                and self.cost_matrix[self.current_node, int(part[1].replace("n", "")), 1] <= self.current_fuel 
                and self.cost_matrix[self.current_node, int(part[1].replace("n", "")), 0] <= (self.max_time_horizon - self.current_time)
            ):
                allowed.append(int(part[1].replace("n", "")))
        return allowed


    def action_masks(self) -> list[bool]:
        mask = [False] * self.num_nodes
        allowed_actions = self.get_allowed_actions()
        for node in allowed_actions:
            mask[node] = True
        return mask


    def _get_obs(self):
        return {
            "current_node": self.current_node,
            "current_fuel": np.array([self.current_fuel], dtype=np.float32),
            "current_time": np.array([self.current_time], dtype=int),
            "visited_satellites": self.visited_satellites.astype(np.int8),
            "action_mask": self.action_masks(),
            "satellites_pos": self.get_satellites_pos(),
        }


    def step(self, action):

        terminated = False
        truncated = False
        info = {}
        reward = 0.0

        # The choosen action is represent by the next node to go
        chosen_node = action

        # Compute the time to go to the next node
        travel_time = int(self.cost_matrix[self.current_node, chosen_node, 0])
        arrival_time = self.current_time + travel_time
        fuel_cost = self.cost_matrix[self.current_node, chosen_node, 1]

        stopforloop = False

        for idx, sat in enumerate(self.satellites):
            if self.visited_satellites[idx] == 1:
                continue

            pos_sat = self.B_v[sat]
            pos_sat_list = sorted(pos_sat.items(), key=lambda x: x[1][1])

            for key, (node_id, node_time) in pos_sat_list:
                node = int(node_id.replace("n", ""))
                if node == chosen_node and arrival_time == node_time:
                    self.visited_satellites[idx] = 1
                    reward = 25.0
                    stopforloop = True
                    break

            if stopforloop:
                break

        if not stopforloop:
            reward = -0.1

        self.route.append(f"{self.current_node} to {chosen_node}")
        self.current_node = chosen_node
        self.current_time += travel_time
        self.current_fuel -= fuel_cost

        mult_fuel_cost = 1
        reward -= fuel_cost * mult_fuel_cost


        if self.current_time > self.max_time_horizon:
            truncated = True
            info["termination"] = "Time horizon reached"
            
        elif self.visited_satellites.sum() == self.num_satellites:
            terminated = True
            info["termination"] = "All satellites maintenaced"

        #print(f"step → t={self.current_time}, done={(terminated or truncated)}, serv={self.serviced_satellites.sum()}")

        return self._get_obs(), reward, terminated, truncated, info

    def render(self, mode="human"):
        print("-----------------------------")
        print("Path :", self.route)
        print("Current fuel :", self.current_fuel)
        print("Current time :", self.current_time)
        print("Maintenaced Satellites :", self.visited_satellites)
        print("-----------------------------\n")


if __name__ == "__main__":

    case_study = "petit_test"
    input_directory = (
        "/Users/nathanclaret/Desktop/thesis/code/data_studycase/"
        + case_study
    )

    env = OOSenv(input_directory)

    observation, _ = env.reset()
    terminated = False
    truncated = False
    total_reward = 0
    step_count = 0

    while not terminated and not truncated:
        # Choose a random action
        action = env.action_space.sample()

        # Skip action if invalid
        masks = env.unwrapped.action_masks()
        if masks[action] == False:
            continue

        # Perform action
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        step_count += 1

        print(f"Step: {step_count} Action: {action}")
        print(f"Observation: {observation}")
        print(f"Reward: {reward}\n")

    print(f"---- Total Reward: {total_reward} ----")
