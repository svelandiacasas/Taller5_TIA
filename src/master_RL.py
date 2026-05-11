import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from collections import deque
import copy



# =====================================================================
# 1. LA MEMORIA 
# =====================================================================
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, next_action):
        self.buffer.append((state, action, reward, next_state, done, next_action))
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones, next_actions = zip(*batch)
        
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones), np.array(next_actions))
    
    def __len__(self):
        return len(self.buffer)

# =====================================================================
# 2. EL AGENTE (Modelos)
# =====================================================================
class MasterRLBase(nn.Module):
    def __init__(self, input_size=4, action_dim=2, hidden_layers=[128, 128], activation_fn=nn.ReLU):
        super().__init__()
        self.action_dim = action_dim
        layers = []
        in_dim = input_size 
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(activation_fn())
            in_dim = h_dim            
        self.net = nn.Sequential(*layers) 
        self.q_values = nn.Linear(in_dim, action_dim)
        
    def forward(self, x):
        features = self.net(x)
        return self.q_values(features)

    def compute_loss(self, states, actions, rewards, next_states, dones, next_actions, target_net, gamma):
        raise NotImplementedError

class MasterDQN(MasterRLBase):
    def compute_loss(self, states, actions, rewards, next_states, dones, next_actions, target_net, gamma):
        current_q = self(states).gather(1, actions)
        with torch.no_grad():
            max_next_q = target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (gamma * max_next_q * (1 - dones))
        return nn.MSELoss()(current_q, target_q)
    
class MasterSARSA(MasterRLBase):
    def compute_loss(self, states, actions, rewards, next_states, dones, next_actions, target_net, gamma):
        # 1. Calculamos el valor Q de la acción actual
        current_q = self(states).gather(1, actions)
        
        with torch.no_grad():
            # 2. SARSA: En lugar de usar .max(), usamos el valor de la ACCIÓN REAL 
            # que el agente tomó en el siguiente paso (next_actions)
            next_q = target_net(next_states).gather(1, next_actions)
            target_q = rewards + (gamma * next_q * (1 - dones))
            
        return nn.MSELoss()(current_q, target_q)

# =====================================================================
# 3. LA EXPLORACIÓN 
# =====================================================================
class EpsilonGreedy:
    def __init__(self, start=1.0, end=0.01, decay=0.995):
        self.epsilon = start
        self.end = end
        self.decay = decay

    def select_action(self, q_values, step, action_dim):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(action_dim)
        else:
            action = q_values.argmax().item()
        self.epsilon = max(self.end, self.epsilon * self.decay)
        return action

# =====================================================================
# 4. EL MOTOR DE ENTRENAMIENTO ENCAPSULADO
# =====================================================================
class MotorRL:
    """Clase principal que orquesta la interacción entre el Agente, el Entorno y la Memoria."""
    def __init__(self, agent, env, optimizer, device, exploration_strategy, 
                 gamma=0.99, batch_size=64, buffer_capacity=10000):
        self.agent = agent
        self.env = env
        self.optimizer = optimizer
        self.device = device
        self.exploration = exploration_strategy
        self.gamma = gamma
        self.batch_size = batch_size
        
        # Inicialización de memoria y clonación de la red objetivo
        self.memory = ReplayBuffer(capacity=buffer_capacity)
        self.target_net = copy.deepcopy(agent).to(device)
        self.target_net.eval()

    def train(self, episodes=300, target_update_freq=10, log_freq=20):
        historial_recompensas = []
        historial_epsilons = []
        historial_resultados = []
        
        print(f"▶ Motor RL Iniciado. Entrenando en {self.device} por {episodes} episodios...")
        start_time = time.time()

        for ep in range(episodes):
            state, _ = self.env.reset()
            total_reward = 0
            
            # Decisión inicial
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_vals = self.agent(state_tensor)
            action = self.exploration.select_action(q_vals, ep, self.agent.action_dim)
            
            while True:
                # Interacción
                next_state, reward, done, truncated, _ = self.env.step(action)
                total_reward += reward
                
                # Planificación de la siguiente acción
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    next_q_vals = self.agent(next_state_tensor)
                next_action = self.exploration.select_action(next_q_vals, ep, self.agent.action_dim)
                
                # Guardar experiencia
                self.memory.push(state, action, reward, next_state, done or truncated, next_action)
                
                # Aprender
                if len(self.memory) >= self.batch_size:
                    self._optimize_step()
                
                # Avanzar
                state = next_state
                action = next_action
                
                if done or truncated:
                    break
                    
            # Sincronización periódica de la red objetivo
            if (ep + 1) % target_update_freq == 0:
                self.target_net.load_state_dict(self.agent.state_dict())
                
            # Log de métricas
            historial_recompensas.append(total_reward)
            historial_epsilons.append(self.exploration.epsilon if hasattr(self, 'exploration') else self.exploration_strategy.epsilon)
            
            # Clasificar el resultado basado en tu rewards_config [3.0, 1.0, -3.0, -10.0]
            if total_reward >= 3.0:
                historial_resultados.append("W")
            elif total_reward > 0 and total_reward < 3.0: # Suele ser 1.0 (Empate)
                historial_resultados.append("D")
            else:
                historial_resultados.append("L")
            
            if (ep + 1) % log_freq == 0: 
                print(f"Episodio {ep+1:03d} | Recompensa Total: {total_reward:5.1f} | Epsilon: {self.exploration.epsilon:.3f}")
    

        print(f"✔ Entrenamiento finalizado en {time.time() - start_time:.2f}s")       
        return {
            "recompensas": historial_recompensas,
            "epsilons": historial_epsilons,
            "resultados": historial_resultados
        }
    def _optimize_step(self):
        """Método interno para manejar el muestreo y el Backpropagation."""
        s, a, r, ns, d, na = self.memory.sample(self.batch_size)
        
        # Envío al dispositivo (GPU/CPU)
        s = torch.FloatTensor(s).to(self.device)
        a = torch.LongTensor(a).unsqueeze(1).to(self.device)
        r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        ns = torch.FloatTensor(ns).to(self.device)
        d = torch.FloatTensor(d).unsqueeze(1).to(self.device)
        na = torch.LongTensor(na).unsqueeze(1).to(self.device)
        
        # Cálculo matemático delegado al tipo de Agente (DQN, SARSA, etc.)
        loss = self.agent.compute_loss(s, a, r, ns, d, na, self.target_net, self.gamma)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# =====================================================================
# 5. EL BLOQUE PRINCIPAL (Ejecución mucho más limpia)
# =====================================================================
"""if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0] 
    action_dim = env.action_space.n            
    
    agente = MasterDQN(input_size=state_dim, action_dim=action_dim).to(device)
    exploracion = EpsilonGreedy(start=1.0, end=0.05, decay=0.999)
    optimizador = optim.Adam(agente.parameters(), lr=0.001)
    
    # Instanciamos el motor
    motor = MotorRL(
        agent=agente, 
        env=env, 
        optimizer=optimizador, 
        device=device,
        exploration_strategy=exploracion,
        batch_size=64
    )
    
    # Ejecutamos el entrenamiento
    historial = motor.train(episodes=1000)
    
    env.close()"""