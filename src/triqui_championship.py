import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt 
import pandas as pd
from triqui import Game as g 
from master_RL import MasterDQN, MasterSARSA, MotorRL, EpsilonGreedy
from triqui_algorithm import Algorithm 
import itertools
import seaborn as sns

class Championship:
    def __init__(self, agents_dict, env_instance, device="cpu"):
        self.agents = agents_dict
        self.agent_names = list(agents_dict.keys())
        self.env = env_instance
        self.device = device
        
        # Estructura de resultados: results[Jugador][Oponente][Iniciativa]
        self.results = {name: {} for name in self.agent_names}
        for p1 in self.agent_names:
            for p2 in self.agent_names:
                if p1 != p2:
                    self.results[p1][p2] = {
                        "Agent_Starts": {"Wins": 0, "Losses": 0, "Draws": 0},
                        "Opponent_Starts": {"Wins": 0, "Losses": 0, "Draws": 0}
                    }
                    
    def run_tournament(self, episodes_per_matchup=200):
        # itertools.combinations evita jugar A vs B y luego B vs A (ahorra 50% de tiempo)
        matchups = list(itertools.combinations(self.agent_names, 2))
        total_matches = len(matchups)
        
        print(f"\nIniciando Torneo: {total_matches} enfrentamientos directos.")
        
        for i, (p1_name, p2_name) in enumerate(matchups):
            print(f"Match {i+1}/{total_matches}: {p1_name} vs {p2_name}")
            agent1 = self.agents[p1_name]
            agent2 = self.agents[p2_name]
            
            # --- MITAD 1: P1 inicia (P1=1, P2=2) ---
            w1, d1, l1 = self._play_match(agent1, agent2, p1_starts=True, episodes=episodes_per_matchup // 2)
            self.results[p1_name][p2_name]["Agent_Starts"]["Wins"] += w1
            self.results[p1_name][p2_name]["Agent_Starts"]["Draws"] += d1
            self.results[p1_name][p2_name]["Agent_Starts"]["Losses"] += l1
            
            # Lo que es victoria para P1, es derrota para P2 (siendo segundo)
            self.results[p2_name][p1_name]["Opponent_Starts"]["Wins"] += l1
            self.results[p2_name][p1_name]["Opponent_Starts"]["Draws"] += d1
            self.results[p2_name][p1_name]["Opponent_Starts"]["Losses"] += w1

            # --- MITAD 2: P2 inicia (P2=1, P1=2) ---
            w2, d2, l2 = self._play_match(agent2, agent1, p1_starts=True, episodes=episodes_per_matchup // 2)
            self.results[p2_name][p1_name]["Agent_Starts"]["Wins"] += w2
            self.results[p2_name][p1_name]["Agent_Starts"]["Draws"] += d2
            self.results[p2_name][p1_name]["Agent_Starts"]["Losses"] += l2
            
            self.results[p1_name][p2_name]["Opponent_Starts"]["Wins"] += l2
            self.results[p1_name][p2_name]["Opponent_Starts"]["Draws"] += d2
            self.results[p1_name][p2_name]["Opponent_Starts"]["Losses"] += w2

    def _play_match(self, p1_model, p2_model, p1_starts, episodes):
        wins, draws, losses = 0, 0, 0
        
        for _ in range(episodes):
            juego = g()  # Instancia PURA del juego (triqui.py), sin nada de RL
            
            # Asignamos IDs: 1 (X) o 2 (O)
            id_p1 = 1 if p1_starts else 2
            id_p2 = 2 if p1_starts else 1
            
            while not juego.game_over:
                # Identificar de quién es el turno
                current_id = juego.current_player
                current_model = p1_model if current_id == id_p1 else p2_model
                
                # 1. Leer el estado directamente del tablero puro y aplanarlo
                state = juego.game_matrix.flatten()
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                # 2. El modelo piensa su jugada
                with torch.no_grad():
                    q_vals = current_model(state_tensor)
                
                # 3. Filtrar solo jugadas válidas (el modelo en torneo NO debe hacer trampas)
                valid_actions = juego.available_positions()
                if len(valid_actions) == 0:
                    break
                    
                valid_indices = [pos[0] * 3 + pos[1] for pos in valid_actions]
                q_vals_valid = q_vals[0, valid_indices]
                action_idx = valid_indices[torch.argmax(q_vals_valid).item()]
                
                # 4. Traducir índice (0-8) a coordenadas (Fila, Columna)
                row = action_idx // 3
                col = action_idx % 3
                
                # 5. Mover la ficha en el tablero puro
                juego._execute_move(row, col, current_id)
                
            # --- FIN DE LA PARTIDA: EVALUAR PARA P1 ---
            if juego.win_condition():
                # En triqui.py, si alguien gana, current_player NO cambia
                if juego.current_player == id_p1:
                    wins += 1
                else:
                    losses += 1
            else:
                # Si terminó sin ganador, es empate
                draws += 1
                
        return wins, draws, losses
    
    
    def get_global_leaderboard(self):
        data = []
        for agent in self.agent_names:
            total_w, total_d, total_l = 0, 0, 0
            for opp in self.agent_names:
                if opp != agent:
                    # Sumamos cuando inicia
                    total_w += self.results[agent][opp]["Agent_Starts"]["Wins"]
                    total_d += self.results[agent][opp]["Agent_Starts"]["Draws"]
                    total_l += self.results[agent][opp]["Agent_Starts"]["Losses"]
                    # Sumamos cuando es segundo
                    total_w += self.results[agent][opp]["Opponent_Starts"]["Wins"]
                    total_d += self.results[agent][opp]["Opponent_Starts"]["Draws"]
                    total_l += self.results[agent][opp]["Opponent_Starts"]["Losses"]
            
            # Puntos: Win = 3, Draw = 1, Loss = 0
            pts = (total_w * 3) + (total_d * 1)
            total_games = total_w + total_d + total_l
            winrate = (total_w / total_games * 100) if total_games > 0 else 0
            drawrate = (total_d / total_games * 100) if total_games > 0 else 0
            
            data.append([agent, pts, total_w, total_d, total_l, winrate, drawrate])
            
        df = pd.DataFrame(data, columns=["Agente", "Puntos", "Wins", "Draws", "Losses", "WinRate", "DrawRate"])
        df = df.sort_values(by=["Puntos", "WinRate"], ascending=[False, False]).reset_index(drop=True)
        # Redondeo para limpieza visual
        df["WinRate"] = df["WinRate"].round(1)
        df["DrawRate"] = df["DrawRate"].round(1)
        return df

    def get_initiative_leaderboards(self):
        data_starting = []
        data_second = []
        
        for agent in self.agent_names:
            # Stats Iniciando
            st_w = sum(self.results[agent][opp]["Agent_Starts"]["Wins"] for opp in self.agent_names if opp != agent)
            st_d = sum(self.results[agent][opp]["Agent_Starts"]["Draws"] for opp in self.agent_names if opp != agent)
            st_l = sum(self.results[agent][opp]["Agent_Starts"]["Losses"] for opp in self.agent_names if opp != agent)
            st_total = st_w + st_d + st_l
            
            st_winrate = (st_w / st_total * 100) if st_total > 0 else 0
            st_drawrate = (st_d / st_total * 100) if st_total > 0 else 0
            st_lossrate = (st_l / st_total * 100) if st_total > 0 else 0
            
            data_starting.append([agent, st_winrate, st_drawrate, st_lossrate])
            
            # Stats Siendo Segundo
            sc_w = sum(self.results[agent][opp]["Opponent_Starts"]["Wins"] for opp in self.agent_names if opp != agent)
            sc_d = sum(self.results[agent][opp]["Opponent_Starts"]["Draws"] for opp in self.agent_names if opp != agent)
            sc_l = sum(self.results[agent][opp]["Opponent_Starts"]["Losses"] for opp in self.agent_names if opp != agent)
            sc_total = sc_w + sc_d + sc_l
            
            sc_winrate = (sc_w / sc_total * 100) if sc_total > 0 else 0
            sc_drawrate = (sc_d / sc_total * 100) if sc_total > 0 else 0
            sc_lossrate = (sc_l / sc_total * 100) if sc_total > 0 else 0
            
            data_second.append([agent, sc_winrate, sc_drawrate, sc_lossrate])
            
        df_start = pd.DataFrame(data_starting, columns=["Agente", "WinRate_Starting", "DrawRate_Starting", "LossRate_Starting"])
        df_start = df_start.sort_values("WinRate_Starting", ascending=False).reset_index(drop=True).round(1)
        
        df_second = pd.DataFrame(data_second, columns=["Agente", "WinRate_Second", "DrawRate_Second", "LossRate_Second"])
        df_second = df_second.sort_values("WinRate_Second", ascending=False).reset_index(drop=True).round(1)
        
        return df_start, df_second

    def plot_heatmaps(self):
        # Crear DataFrames numéricos para los Heatmaps (usaremos Score = Win + 0.5*Draw para el color)
        matrix_start = pd.DataFrame(index=self.agent_names, columns=self.agent_names, dtype=float)
        matrix_second = pd.DataFrame(index=self.agent_names, columns=self.agent_names, dtype=float)
        
        # Matrices para el texto interno (W / D / L)
        annot_start = pd.DataFrame(index=self.agent_names, columns=self.agent_names)
        annot_second = pd.DataFrame(index=self.agent_names, columns=self.agent_names)
        
        for p1 in self.agent_names:
            for p2 in self.agent_names:
                if p1 == p2:
                    matrix_start.loc[p1, p2] = np.nan
                    matrix_second.loc[p1, p2] = np.nan
                    annot_start.loc[p1, p2] = "-"
                    annot_second.loc[p1, p2] = "-"
                else:
                    # Cuando P1 (Fila) INICIA
                    res_p1 = self.results[p1][p2]["Agent_Starts"]
                    tot_p1 = res_p1["Wins"] + res_p1["Draws"] + res_p1["Losses"]
                    if tot_p1 > 0:
                        score_p1 = (res_p1["Wins"] + 0.5 * res_p1["Draws"]) / tot_p1
                        matrix_start.loc[p1, p2] = score_p1 * 100
                    annot_start.loc[p1, p2] = f"W:{res_p1['Wins']}\nD:{res_p1['Draws']}\nL:{res_p1['Losses']}"
                        
                    # Cuando P1 (Fila) ES SEGUNDO
                    res_p2 = self.results[p1][p2]["Opponent_Starts"]
                    tot_p2 = res_p2["Wins"] + res_p2["Draws"] + res_p2["Losses"]
                    if tot_p2 > 0:
                        score_p2 = (res_p2["Wins"] + 0.5 * res_p2["Draws"]) / tot_p2
                        matrix_second.loc[p1, p2] = score_p2 * 100
                    annot_second.loc[p1, p2] = f"W:{res_p2['Wins']}\nD:{res_p2['Draws']}\nL:{res_p2['Losses']}"

        # Graficar ambos Heatmaps lado a lado
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        sns.heatmap(matrix_start, annot=annot_start, fmt="", cmap="RdYlGn", vmin=0, vmax=100, 
                    ax=axes[0], cbar_kws={'label': 'Efectividad % (Wins + 0.5*Empates)'})
        axes[0].set_title("Resultados cuando el Agente (Fila) INICIA", fontweight='bold')
        axes[0].set_ylabel("Agente Evaluado")
        axes[0].set_xlabel("Oponente")
        
        sns.heatmap(matrix_second, annot=annot_second, fmt="", cmap="RdYlGn", vmin=0, vmax=100, 
                    ax=axes[1], cbar_kws={'label': 'Efectividad % (Wins + 0.5*Empates)'})
        axes[1].set_title("Resultados cuando el Agente (Fila) es SEGUNDO", fontweight='bold')
        axes[1].set_ylabel("Agente Evaluado")
        axes[1].set_xlabel("Oponente")
        
        plt.tight_layout()
        plt.show()