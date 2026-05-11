from triqui import Game as g 
import numpy as np
import pandas as pd
import time

class Simulator:
    def __init__(self, n=1, verbose=False):
        self.n = n
        self.verbose = verbose
        self.resultados = np.zeros(n, dtype=np.int8) 
        self.movs_hist = np.zeros(n, dtype=np.int8)  

    def play(self):
        start_time = time.time()
        for i in range(self.n):
            juego = g() 
            
            if self.verbose:
                print(f"\n" + "═"*30)
                print(f"🎮 PARTIDA {i+1} INICIADA")
                print("═"*30)
            
            while not juego.game_over:
                libres = juego.available_positions()
                if len(libres) == 0: break
                
                # Selección de jugada aleatoria
                idx = np.random.randint(0, len(libres))
                pos = libres[idx]
                
                jugador_actual = juego.current_player
                if jugador_actual == 1:
                    juego.play1(pos[0], pos[1])
                else:
                    juego.play2(pos[0], pos[1])
                
                # --- EFECTO VERBOSE: TABLERO EN CADA JUGADA ---
                if self.verbose:
                    print(f"\n👤 P{jugador_actual} movió a: ({pos[0]}, {pos[1]})")
                    print(juego.get_game_matrix()) # Muestra la matriz tras el movimiento

            # Registro de estadísticas finales de la partida
            self.movs_hist[i] = juego.turns_played
            self.resultados[i] = self._get_winner(juego)
            
            if self.verbose:
                resultado_texto = "🤝 Empate" if self.resultados[i] == 0 else f"🏆 Ganador: P{self.resultados[i]}"
                print(f"\n{resultado_texto}")
                print(f"Total movimientos: {juego.turns_played}")
            
            # Contador de progreso para simulaciones masivas (solo si verbose es False)[cite: 5]
            if not self.verbose and (i + 1) % 100000 == 0:
                 print(f"⏳ {i + 1} partidas procesadas...")

        end_time = time.time()
        print(f"\n⏱️ Tiempo total de simulación: {end_time - start_time:.2f}s")
        self.print_summary_dataframe()

    def _get_winner(self, juego):
        m = juego.get_game_matrix()
        for p in [1, 2]:
            target = 3 * p
            if np.any(m.sum(axis=0) == target) or np.any(m.sum(axis=1) == target) or \
               m.diagonal().sum() == target or np.fliplr(m).diagonal().sum() == target:
                return p
        return 0 

    def print_summary_dataframe(self):
        df = pd.DataFrame({
            "ganador_id": self.resultados,
            "movimientos": self.movs_hist
        })
        
        map_res = {0: "Empate", 1: "P1 (X)", 2: "P2 (O)"}
        df["ganador"] = df["ganador_id"].map(map_res)

        counts = df['ganador'].value_counts()
        winrate = (df['ganador'].value_counts(normalize=True) * 100)

        stats_df = pd.DataFrame({
            'Total Partidas': counts,
            'Winrate (%)': winrate.map('{:.2f}%'.format)
        })

        print("\n" + "="*55)
        print(f"📊 REPORTE ESTADÍSTICO DE {self.n:,} SIMULACIONES")
        print("="*55)
        print(stats_df)
        
        print("\n--- Eficiencia por Turnos ---")
        avg_movs = df.groupby('ganador')['movimientos'].mean().round(2)
        print(avg_movs.to_frame('Promedio Movimientos'))
        print("="*55)

if __name__ == "__main__":
    simulacion = Simulator(n=1000000, verbose=False)
    simulacion.play()