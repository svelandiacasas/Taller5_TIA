import numpy as np

class Game:
    def __init__(self):
        self.game_matrix = np.zeros((3, 3), dtype=int)
        self.current_player = 1 
        self.turns_played = 0
        self.game_over = False

    def _execute_move(self, row, col, player_id):
        if self.game_over or self.game_matrix[row, col] != 0:
            # print("🛑 El juego terminó.")
            return False

        if not (0 <= row <= 2 and 0 <= col <= 2):
            # print(f"❌ Error: ({row}, {col}) fuera de rango.")
            return False

        if self.current_player != player_id:
            # print(f"🚫 Turno del Jugador {self.current_player}")
            return False

        if self.game_matrix[row, col] != 0:
            # print(f"⚠️ Posición ({row}, {col}) ocupada.")
            return False

        self.game_matrix[row, col] = player_id
        self.turns_played += 1
        
        if self.win_condition():
            self.game_over = True
            return True
        
        if self.turns_played == 9:
            self.game_over = True
            return True

        self.current_player = 2 if self.current_player == 1 else 1

            
        return True

    def play1(self, row, col):
        return self._execute_move(row, col, 1)

    def play2(self, row, col):
        return self._execute_move(row, col, 2)

    def _auto_last_move(self):
        # Uso de np.argwhere para mantener coherencia en NumPy
        pos = np.argwhere(self.game_matrix == 0)[0]
        self._execute_move(pos[0], pos[1], self.current_player)

    def win_condition(self):
        # Validación pura en NumPy: revisa sumas de filas, columnas y diagonales
        m = self.game_matrix
        p = self.current_player
        
        # Revisar filas (axis 1) y columnas (axis 0)
        if np.any(np.all(m == p, axis=1)) or np.any(np.all(m == p, axis=0)):
            return True
        # Revisar diagonales
        if np.all(np.diag(m) == p) or np.all(np.diag(np.fliplr(m)) == p):
            return True
        return False

    def available_positions(self):
        # Retorna un array de NumPy directamente
        return np.argwhere(self.game_matrix == 0)
    
    def get_game_matrix(self):
        return self.game_matrix
    
    def get_winner(self):
        m = self.game_matrix
        for p in [1, 2]:
            if np.any(np.all(m == p, axis=1)) or \
            np.any(np.all(m == p, axis=0)) or \
            np.all(np.diag(m) == p) or \
            np.all(np.diag(np.fliplr(m)) == p):
                return p
        return 0