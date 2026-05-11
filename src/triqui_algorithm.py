from triqui import Game as g 
import numpy as np
import pandas as pd

class Algorithm:
    def __init__(self, juego):        
        self.juego = juego
        self.strat = 0
    def get_available(self):
        return self.juego.available_positions()

    def get_game_matrix(self):
        return self.juego.get_game_matrix()
    
    def automovement(self, player_id):
        available = self.get_available()
        matrix_sim = self.get_game_matrix().copy() 
        opponent_id = 3 - player_id 

        def victory(matriz, jugador):
            if np.any(np.all(matriz == jugador, axis=1)) or np.any(np.all(matriz == jugador, axis=0)):
                return True
            if np.all(np.diag(matriz) == jugador) or np.all(np.diag(np.fliplr(matriz)) == jugador):
                return True
            return False

        # --- FASE 1: Victoria propia ---
        for pos in available:
            r, c = pos[0], pos[1]
            matrix_sim[r, c] = player_id 
            if victory(matrix_sim, player_id):
                return (r, c)            
            matrix_sim[r, c] = 0 

        # --- FASE 2: Bloqueo crítico ---
        for pos in available:
            r, c = pos[0], pos[1]
            matrix_sim[r, c] = opponent_id 
            if victory(matrix_sim, opponent_id):
                return (r, c)            
            matrix_sim[r, c] = 0 

        return None
    
    def randommovement(self):
        available = self.get_available()
        if len(available) == 0:
            return None
        choice = available[np.random.randint(0, len(available))]
        return (choice[0], choice[1])   
    
    def play1(self):
        t = self.juego.turns_played
        idx = t // 2  
        
        # --- 1. Caso Inicial (Turno 0) ---
        if t == 0:
            self.strat = 0
            return (1, 1)
        # ---  MOVIMIENTO FORZADO GLOBAL ---
        forced = self.automovement(player_id=1)
        if forced is not None:
            return forced
        
        if t == 2:
            move = self.s_t2()
            if move: return move
        elif t == 4:
            move = self.s_t4()
            if move: return move
        elif t == 8:
            pos = self.get_available()[0]
            return (pos[0], pos[1])
        else:
            # Movimiento aleatorio como fallback ya que no afecta la victoria ni la derrota dado a que el automovement ya se encarga de eso.
            return self.randommovement()
                
    def s_t2(self):
        game_matrix = self.get_game_matrix()
        corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
        edges = [(0, 1), (1, 0), (1, 2), (2, 1)]
        
        # Caso Esquina
        for r, c in corners:
            if game_matrix[r, c] == 2:
                self.strat = 1
                return (r, 2 - c)
                
        # Caso Arista
        for r, c in edges:
            if game_matrix[r, c] == 2:
                self.strat = 2
                adj_corners = {
                        (0, 1): [(0, 0), (0, 2)],
                        (2, 1): [(2, 0), (2, 2)],
                        (1, 0): [(0, 0), (2, 0)],
                        (1, 2): [(0, 2), (2, 2)] 
                    }
                pick = adj_corners[(r, c)][np.random.randint(0, 2)]                    
                return (pick[0], pick[1])
        #Fallback 
        return self.randommovement()
    
    def s_t4(self):
        game_matrix = self.get_game_matrix()        
        if self.strat == 2:
            #Continua el caso de aristas
            edges = [(0, 1), (1, 0), (1, 2), (2, 1)]
            p2_edge = next((r, c) for r, c in edges if game_matrix[r, c] == 2)
            safe_corners_map = {
                (0, 1): [(2, 0), (2, 2)], 
                (2, 1): [(0, 0), (0, 2)], 
                (1, 0): [(0, 2), (2, 2)], 
                (1, 2): [(0, 0), (2, 0)]  
            }            
            options = safe_corners_map[p2_edge]
            pick = options[np.random.randint(0, 2)]            
            return (pick[0], pick[1])
        #Fallback 
        return self.randommovement()
    """para el caso s_t6 ya o gana automaticamente en la strat2 con el automovement 
    o con el automovement en las demas estrategias garantiza el empate entre random y bloqueos
    No se requiere mas algoritmos para estos casos."""

    def play2(self):
        t = self.juego.turns_played
        idx = t // 2 
        # --- 1. Caso Inicial (Turno 1) ---
        if t == 1:
            return self.s_t1()
        
        if t > 1:
            forced = self.automovement(player_id=2)
            if forced is not None:
                return forced
        
        if t == 3:
            move = self.s_t3()
            if move: return move
        elif t == 5:
            move = self.s_t5()
            if move: return move
        
        else:
            # Movimiento aleatorio como fallback ya que no afecta la victoria ni la derrota dado a que el automovement ya se encarga de eso.
            return self.randommovement()
                
    def s_t1(self):
        game_matrix = self.get_game_matrix()        
        # Caso 1: El centro está libre (El rival jugó Esquina o Arista)
        if game_matrix[1, 1] == 0:
            corners = [(0, 0), (0, 2), (2, 0), (2, 2)]            
            # Verificamos si la ficha del rival está en alguna esquina
            jugo_esquina = any(game_matrix[r, c] == 1 for r, c in corners)            
            if jugo_esquina:
                self.strat = 4 
            else:
                self.strat = 5  # Por descarte, es Arista
            return (1, 1)
        # Caso 2: El rival tomó el centro
        else:
            self.strat = 3 
            corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
            pick = corners[np.random.randint(0, 4)]
            return (pick[0], pick[1])
        
    def s_t3(self):
        game_matrix = self.get_game_matrix() 
        corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
        edges = [(0, 1), (1, 0), (1, 2), (2, 1)]      
        
        
        #Caso Centro inicial del rival y esquina de misma diagonal
        if self.strat == 3:
            my_r, my_c = np.argwhere(game_matrix == 2)[0]
            p1_pieces = np.argwhere(game_matrix == 1)
            riv_r, riv_c = [pos for pos in p1_pieces if not (pos[0] == 1 and pos[1] == 1)][0]
            
            if riv_r == 2 - my_r and riv_c == 2 - my_c:            
                free_corners = [(r, c) for r, c in corners if game_matrix[r, c] == 0]
                if free_corners:
                    pick = free_corners[np.random.randint(0, len(free_corners))]
                    return (pick[0], pick[1])
        #Esta estrategia garantiza empate con los demas movimientos bloqueantes o randomizados.
    
        """Cualquier otro movimiento del rival en el turno previo donde el inicio el juego con centro, consiste en un ataque,
        por lo que el autobloqueo actua y se que entre bloqueos y random se garantiza el empate.
        """
        
        if self.strat == 4:
            #Indica que el rival inicio en esquina y jugamos centro
            p1_pieces = np.argwhere(game_matrix == 1)
            r1, c1 = p1_pieces[0]
            r2, c2 = p1_pieces[1]
            #Caso Esquina inicial del rival y esquina de misma diagonal
            #Verificar que las diagonales del rival no comparten fila o columna.            
            if (r1, c1) in corners and (r2, c2) in corners: 
                if r1 + r2 == 2 and c1 + c2 == 2:
                    # En una matriz 3x3, las coordenadas de esquinas opuestas siempre suman 2
                    edges = [(0, 1), (1, 0), (1, 2), (2, 1)]            
                    #Todas las aristas estan vacias por las jugadas previas, jugar cualquiera es la unica forma de no perder, el automovement se encarga del resto.
                    pick = edges[np.random.randint(0, len(edges))]
                    return (pick[0], pick[1])

            #Caso Esquina inicial del rival y arista no adyacente
            #Indica que el rival inicio sin el centro y jugamos centro
            #Se verifica que el rival tenga su jugada en una arista no adyacente a su esquina.
            else:
                cr, cc = (r1, c1) if (r1, c1) in corners else (r2, c2)
                er, ec = (r2, c2) if (r1, c1) in corners else (r1, c1)
                #Se juega en la esquina adyacente a la arista del rival y que comparte o fila o columna de la esquina del rival.
                if cr != er and cc != ec:
                    if er in (0, 2):
                        return (er, cc)
                    else:
                        return (cr, ec)
                    #El automovement y el random se encarga del empate.
            """Cualquier otro movimiento con inicio de esquina del rival, implica un bloqueo forzado por lo que el automovement se encarga. 
            """
        if self.strat == 5: 
            #Indica que el rival inicio en arista y jugamos centro
            p1_pieces = np.argwhere(game_matrix == 1)
            r1, c1 = p1_pieces[0]
            r2, c2 = p1_pieces[1]
            #Caso Arista inicial del rival y arista de misma fila o columna del rival
            #Se verifica que el rival tenga sus dos jugadas en una misma fila o columna de aristas.
            if (r1, c1) in edges and (r2, c2) in edges:
                if r1 == r2 or c1 == c2: 
                    #Todas las esquinas estan vacias por las jugadas previas, jugar cualquiera garantiza victoria continuando el algoritmo.
                    pick = corners[np.random.randint(0, len(corners))]
                    return (pick[0], pick[1])
                
            #Caso Arista inicial del rival y esquina no adyacente del rival
            #Se verifica que el rival tenga su jugada en una arista no adyacente a su esquina.
            else:
                cr, cc = (r1, c1) if (r1, c1) in corners else (r2, c2)
                er, ec = (r2, c2) if (r1, c1) in corners else (r1, c1)
                #Se juega en la esquina adyacente a la arista del rival y que comparte o fila o columna de la esquina del rival.
                if cr != er and cc != ec:
                    if er in (0, 2):
                        return (er, cc)
                    else:
                        return (cr, ec)
        #Fallback 
        return self.randommovement()
            
    def s_t5(self):
        if self.strat == 5:#Aca solo se llega desde el Caso Arista inicial del rival y arista de misma fila o columna del rival, dado a que el otro genera bloqueo automatico en este turno
            #Debe jugar en la esquina que comparte fila o columna con la esquina propia y que no comparte fila o columna con las aristas del rival, asi se garantiza victoria automatica en el siguiente turno.
            game_matrix = self.get_game_matrix()
            p2_pieces = np.argwhere(game_matrix == 2)            
            er, ec = [pos for pos in p2_pieces if not (pos[0] == 1 and pos[1] == 1)][0]
            all_corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
            for cr, cc in all_corners:
                # Condición A: La esquina objetivo debe estar vacía
                if game_matrix[cr, cc] != 0:
                    continue
                # Condición B: Debe compartir fila O columna (no estar en diagonal)
                if (cr == er and cc != ec) or (cc == ec and cr != er):
                    arista_media_r = (er + cr) // 2
                    arista_media_c = (ec + cc) // 2
                    
                    if game_matrix[arista_media_r, arista_media_c] == 0:
                        self.juego.play2(cr, cc)
                        return (cr, cc)
        return self.randommovement()
            
                
        """Con esto se garantiza que el automovement en el siguiente turno gane automaticamente luego de tender la trampa"""