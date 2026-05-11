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

class Train:
    def __init__(self, seed = None, rewards_config=[3.0, 1.0, -3.0, -10.0], device="cpu"):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        self.observation_space = type('obj', (object,), {'shape': (9,)}) 
        self.action_space = type('obj', (object,), {'n': 9})
        self.juego = None
        self.agent_role = 2 
        self.device = device
        self.rewards = np.array(rewards_config, dtype=np.float32)
    
    def reset(self):
        self.juego = g() 
        self.history = [] 
        self.starter = None
        self.agent_role = 1 if self.agent_role == 2 else 2
        
        # Si el oponente inicia (role 1)
        if self.agent_role == 2:
            self.starter = "opponent"
            row, col = self._play_opponent(role=1) 
            self._save_movement(row, col, "opponent")
        else:
            self.starter = "agent"
        return self._get_state(), {}
    
    def _get_state(self):
        return self.juego.get_game_matrix().flatten().astype(np.float32)

    def _save_movement(self, row, col, player):
        position_type = self._classify_movement(row, col)
        self.history.append({
            "turn": len(self.history) + 1,
            "player": player,
            "position": position_type,
            "coordinate": (int(row), int(col))
        })
        
    def _classify_movement(self, row, col):
        """Clasifica entre centro, esquina y arista."""
        if row == 1 and col == 1:
            return "center"
        elif (row, col) in [(0, 0), (0, 2), (2, 0), (2, 2)]:
            return "corner"
        else:
            return "edge"

    def step(self, action):
        """Step regula la logica de turnos del agente y el rival y asigna las recompensas"""
        row, col = action // 3, action % 3
        #Esta forma es un truco que permite jugar con numeros del  al 8
        
        #Diccionario de informacion
        info = {
            "starter": self.starter,
            "history": self.history,
            "result": "in_progress"
        }
        
        # 1. Movimiento Ilegal (Penalty)
        if self.juego.get_game_matrix()[row, col] != 0:
            info["result"] = "illegal_move"
            return self._get_state(), self.rewards[3], True, False, info
        
        # Turno Agente
        if self.agent_role == 1: self.juego.play1(row, col)
        else: self.juego.play2(row, col)
        self._save_movement(row, col, "agent")
        
        #Verifica el fin del juego por el agente y asigna recompensas
        if self.juego.game_over:
            ganador = self.juego.get_winner()  
            if ganador == self.agent_role:
                reward = self.rewards[0]  # Ganó el agente
                info["result"] = "agent_win"
            else:
                reward = self.rewards[1]  # Empate (ganador es 0)
                info["result"] = "draw"
            return self._get_state(), reward, True, False, info
        
        # Turno Oponente
        opp_role = 2 if self.agent_role == 1 else 1
        opp_row, opp_col = self._play_opponent(role=opp_role)
        self._save_movement(opp_row, opp_col, "opponent")
        
        # Verifica el fin del juego por el oponente y asigna recompensas
        if self.juego.game_over:
            ganador = self.juego.get_winner()  
            if ganador == opp_role:
                reward = self.rewards[2]  # Perdió el agente
                info["result"] = "opponent_win"
            else:
                reward = self.rewards[1]  # Empate (ganador es 0)
                info["result"] = "draw"
            return self._get_state(), reward, True, False, info
        #state, reward, terminated, truncated, info
        return self._get_state(), 0.0, False, False, info
    
    def _play_opponent(self, role):
        # 1. Caso: No hay oponente (Movimiento Aleatorio)
        if self.opponent is None:
            libres = self.juego.available_positions()
            pos = libres[np.random.randint(0, len(libres))]
            row, col = pos[0], pos[1]
            
        # 2. Caso: El oponente es una Red Neuronal (MasterDQN / Maestro)
        elif isinstance(self.opponent, nn.Module):
            # Extraemos el estado y lo convertimos a Tensor
            state = self._get_state()
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            self.opponent.eval() # Asegurar que no esté en modo entrenamiento
            with torch.no_grad():
                q_values = self.opponent(state_t).squeeze()
            
            # Elegir la mejor acción de las disponibles
            libres_idx = [r*3 + c for r, c in self.juego.available_positions()]
            # Máscara para evitar jugadas ilegales de la IA oponente
            action = libres_idx[q_values[libres_idx].argmax().item()]
            row, col = action // 3, action % 3
            
        # 3. Caso: El oponente es una función (Algoritmo)
        else:
            row, col = self.opponent(self.juego)
            
        # Ejecutar el movimiento en el juego
        if role == 1: self.juego.play1(row, col)
        else: self.juego.play2(row, col)
        
        return row, col
    
    def evaluate_agent(agente, env_eval, episodes=500, device="cpu"):
        agente.eval()
        
        # Función auxiliar para crear la estructura vacía de movimientos (del 1 al 4)
        def create_move_struct():
            return {
                f"Move_{i}": {
                    "center": {"Wins": 0, "Losses": 0, "Draws": 0, "Total": 0},
                    "corner": {"Wins": 0, "Losses": 0, "Draws": 0, "Total": 0},
                    "edge":   {"Wins": 0, "Losses": 0, "Draws": 0, "Total": 0}
                } for i in range(1, 5)
            }

        # 1. Diccionario base: Ahora Moves_Analysis se divide PRIMERO por quién inició
        res = {
            "Wins": 0, "Losses": 0, "Draws": 0, "Illegals": 0,
            "Agent_Starts": {"Wins": 0, "Losses": 0, "Draws": 0, "Total": 0},
            "Opponent_Starts": {"Wins": 0, "Losses": 0, "Draws": 0, "Total": 0},
            "Moves_Analysis": {
                "Agent_Starts": create_move_struct(),
                "Opponent_Starts": create_move_struct()
            }
        }
        
        with torch.no_grad():
            for _ in range(episodes):
                state, _ = env_eval.reset()
                done = False
                info_final = None 
                
                while not done:
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action = agente(state_t).argmax().item()
                    state, reward, done, _, info = env_eval.step(action)
                    
                    if done:
                        info_final = info
                        result = info_final["result"]
                        
                        # --- A. Métricas Generales ---
                        if result == "agent_win": res["Wins"] += 1
                        elif result == "opponent_win": res["Losses"] += 1
                        elif result == "draw": res["Draws"] += 1
                        elif result == "illegal_move": res["Illegals"] += 1
                        
                        if result != "illegal_move":
                            
                            # --- B. Identificar Iniciativa ---
                            starter = info_final.get("starter", "agent")
                            init_key = "Agent_Starts" if starter == "agent" else "Opponent_Starts"
                            
                            # Sumar a métricas globales de iniciativa
                            res[init_key]["Total"] += 1
                            if result == "agent_win": res[init_key]["Wins"] += 1
                            elif result == "opponent_win": res[init_key]["Losses"] += 1
                            elif result == "draw": res[init_key]["Draws"] += 1

                            # --- C. Análisis de Movimientos (Separado por iniciativa) ---
                            historial = info_final.get("history", [])
                            agent_moves = [m for m in historial if m["player"] == "agent"]
                            
                            # Iteramos sobre sus jugadas y las guardamos DENTRO de su respectiva iniciativa
                            for i, mov in enumerate(agent_moves[:4]):
                                move_level = f"Move_{i+1}"
                                pos_type = mov["position"] 
                                
                                res["Moves_Analysis"][init_key][move_level][pos_type]["Total"] += 1
                                if result == "agent_win": 
                                    res["Moves_Analysis"][init_key][move_level][pos_type]["Wins"] += 1
                                elif result == "opponent_win": 
                                    res["Moves_Analysis"][init_key][move_level][pos_type]["Losses"] += 1
                                elif result == "draw": 
                                    res["Moves_Analysis"][init_key][move_level][pos_type]["Draws"] += 1

        agente.train()
        
        # 2. Formateo a Porcentajes Finales
        processed_metrics = {
            "Wins": (res["Wins"] / episodes) * 100,
            "Losses": (res["Losses"] / episodes) * 100,
            "Draws": (res["Draws"] / episodes) * 100,
            "Illegals": (res["Illegals"] / episodes) * 100
        }
        
        def calc_pct(subdict):
            total = subdict["Total"]
            if total == 0: 
                return {"WinRate": 0.0, "LossRate": 0.0, "DrawRate": 0.0, "Usage": 0.0}
            return {
                "WinRate": (subdict["Wins"] / total) * 100,
                "LossRate": (subdict["Losses"] / total) * 100,
                "DrawRate": (subdict["Draws"] / total) * 100,
                "Usage": (total / episodes) * 100
            }

        processed_metrics["Agent_Starts"] = calc_pct(res["Agent_Starts"])
        processed_metrics["Opponent_Starts"] = calc_pct(res["Opponent_Starts"])
        
        # Procesar los porcentajes respetando la separación por iniciativa
        processed_metrics["Moves_Analysis"] = {"Agent_Starts": {}, "Opponent_Starts": {}}
        
        for ikey in ["Agent_Starts", "Opponent_Starts"]:
            for move_level, pos_data in res["Moves_Analysis"][ikey].items():
                processed_metrics["Moves_Analysis"][ikey][move_level] = {
                    "center": calc_pct(pos_data["center"]),
                    "corner": calc_pct(pos_data["corner"]),
                    "edge": calc_pct(pos_data["edge"])
                }
                
        return processed_metrics
        
    def show_plots(df_metricas, df_recompensas):
        episodes = df_metricas['Episodio']

        # ==========================================================
        # Gráficas 1 y 2: Recompensas y Rendimiento General
        # ==========================================================
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
        
        ax1.plot(df_recompensas['Recompensa'].rolling(200).mean(), color='orange', label='Media Móvil Reward')
        ax1.set_title("Progreso de Recompensa Promedio")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.plot(episodes, df_metricas['Wins'], label='Victorias', color='green', linewidth=2)
        ax2.plot(episodes, df_metricas['Draws'], label='Empates', color='blue', linewidth=2)
        ax2.plot(episodes, df_metricas['Losses'], label='Derrotas', color='red', linewidth=2)
        ax2.plot(episodes, df_metricas['Illegals'], label='Inválidas', color='black', linestyle='--')
        ax2.set_title("Evolución del Rendimiento del Agente")
        ax2.set_ylabel("Porcentaje (%)")
        ax2.set_ylim(-5, 105)
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.legend()
        plt.tight_layout()
        plt.show()

        # ==========================================================
        # Gráfica 3: Impacto de la Iniciativa
        # ==========================================================
        winrate_agente_inicia = df_metricas['Agent_Starts'].apply(lambda x: x['WinRate'])
        winrate_oponente_inicia = df_metricas['Opponent_Starts'].apply(lambda x: x['WinRate'])
        
        plt.figure(figsize=(10, 4))
        plt.plot(episodes, winrate_agente_inicia, label='Gana cuando Inicia', color='purple', linewidth=2)
        plt.plot(episodes, winrate_oponente_inicia, label='Gana cuando es Segundo', color='brown', linewidth=2)
        plt.title("Ventaja Táctica: Porcentaje de Victoria según Iniciativa")
        plt.ylabel("Win Rate (%)")
        plt.ylim(-5, 105)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # ==========================================================
        # Gráfica 4: Análisis W/D/L de la Segunda Jugada (Move 2)
        # Separando si inició el agente o si inició el oponente
        # ==========================================================
        
        # Extraemos métricas cuando el Agente INICIÓ
        w_ctr_a = df_metricas['Moves_Analysis'].apply(lambda x: x['Agent_Starts']['Move_2']['center']['WinRate'])
        d_ctr_a = df_metricas['Moves_Analysis'].apply(lambda x: x['Agent_Starts']['Move_2']['center']['DrawRate'])
        l_ctr_a = df_metricas['Moves_Analysis'].apply(lambda x: x['Agent_Starts']['Move_2']['center']['LossRate'])

        w_crn_a = df_metricas['Moves_Analysis'].apply(lambda x: x['Agent_Starts']['Move_2']['corner']['WinRate'])
        d_crn_a = df_metricas['Moves_Analysis'].apply(lambda x: x['Agent_Starts']['Move_2']['corner']['DrawRate'])
        l_crn_a = df_metricas['Moves_Analysis'].apply(lambda x: x['Agent_Starts']['Move_2']['corner']['LossRate'])

        w_edg_a = df_metricas['Moves_Analysis'].apply(lambda x: x['Agent_Starts']['Move_2']['edge']['WinRate'])
        d_edg_a = df_metricas['Moves_Analysis'].apply(lambda x: x['Agent_Starts']['Move_2']['edge']['DrawRate'])
        l_edg_a = df_metricas['Moves_Analysis'].apply(lambda x: x['Agent_Starts']['Move_2']['edge']['LossRate'])

        # Extraemos métricas cuando el Oponente INICIÓ
        w_ctr_o = df_metricas['Moves_Analysis'].apply(lambda x: x['Opponent_Starts']['Move_2']['center']['WinRate'])
        d_ctr_o = df_metricas['Moves_Analysis'].apply(lambda x: x['Opponent_Starts']['Move_2']['center']['DrawRate'])
        l_ctr_o = df_metricas['Moves_Analysis'].apply(lambda x: x['Opponent_Starts']['Move_2']['center']['LossRate'])

        w_crn_o = df_metricas['Moves_Analysis'].apply(lambda x: x['Opponent_Starts']['Move_2']['corner']['WinRate'])
        d_crn_o = df_metricas['Moves_Analysis'].apply(lambda x: x['Opponent_Starts']['Move_2']['corner']['DrawRate'])
        l_crn_o = df_metricas['Moves_Analysis'].apply(lambda x: x['Opponent_Starts']['Move_2']['corner']['LossRate'])

        w_edg_o = df_metricas['Moves_Analysis'].apply(lambda x: x['Opponent_Starts']['Move_2']['edge']['WinRate'])
        d_edg_o = df_metricas['Moves_Analysis'].apply(lambda x: x['Opponent_Starts']['Move_2']['edge']['DrawRate'])
        l_edg_o = df_metricas['Moves_Analysis'].apply(lambda x: x['Opponent_Starts']['Move_2']['edge']['LossRate'])

        # Creamos una cuadrícula de 2 filas x 3 columnas
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
        fig.suptitle("Análisis de Segunda Jugada (Move 2): Efectividad W/D/L por Posición", fontsize=16)

        # FILA 1: Cuando el Agente Inicia (Turno 3 del juego real)
        axes[0, 0].plot(episodes, w_ctr_a, 'g-', label='WinRate')
        axes[0, 0].plot(episodes, d_ctr_a, 'b--', label='DrawRate')
        axes[0, 0].plot(episodes, l_ctr_a, 'r:', label='LossRate')
        axes[0, 0].set_title("Si Agente Inició -> Jugó Centro")
        axes[0, 0].grid(True, alpha=0.3); axes[0, 0].legend()

        axes[0, 1].plot(episodes, w_crn_a, 'g-')
        axes[0, 1].plot(episodes, d_crn_a, 'b--')
        axes[0, 1].plot(episodes, l_crn_a, 'r:')
        axes[0, 1].set_title("Si Agente Inició -> Jugó Esquina")
        axes[0, 1].grid(True, alpha=0.3)

        axes[0, 2].plot(episodes, w_edg_a, 'g-')
        axes[0, 2].plot(episodes, d_edg_a, 'b--')
        axes[0, 2].plot(episodes, l_edg_a, 'r:')
        axes[0, 2].set_title("Si Agente Inició -> Jugó Arista")
        axes[0, 2].grid(True, alpha=0.3)

        # FILA 2: Cuando el Oponente Inicia (Turno 4 del juego real)
        axes[1, 0].plot(episodes, w_ctr_o, 'g-')
        axes[1, 0].plot(episodes, d_ctr_o, 'b--')
        axes[1, 0].plot(episodes, l_ctr_o, 'r:')
        axes[1, 0].set_title("Si Oponente Inició -> Jugó Centro")
        axes[1, 0].set_xlabel("Episodios")
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(episodes, w_crn_o, 'g-')
        axes[1, 1].plot(episodes, d_crn_o, 'b--')
        axes[1, 1].plot(episodes, l_crn_o, 'r:')
        axes[1, 1].set_title("Si Oponente Inició -> Jugó Esquina")
        axes[1, 1].set_xlabel("Episodios")
        axes[1, 1].grid(True, alpha=0.3)

        axes[1, 2].plot(episodes, w_edg_o, 'g-')
        axes[1, 2].plot(episodes, d_edg_o, 'b--')
        axes[1, 2].plot(episodes, l_edg_o, 'r:')
        axes[1, 2].set_title("Si Oponente Inició -> Jugó Arista")
        axes[1, 2].set_xlabel("Episodios")
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()       
        
    