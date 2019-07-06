import torch
import torch as nn
import torch.nn.functional as F

class NN(nn.Module):
    """Clase generica que implementa una red neuronal feed-forward con dos capas ocultas y activacion ReLU."""
    
    def __init__(self, n_input, n_output, seed, fc1_units=64, fc2_units=64):
        """Inicializar parametros y construir red neuronal
        Params
        ======
            n_input (int):   numero de neuronas de la capa de entrada
            n_output (int):  numero de neuronas de la capa de salida
            seed (int):      semilla del generador de numeros aleatorio
            fc1_units (int): numero de neuronas en la primera capa oculta
            fc2_units (int): numero de neuronas en la segunda capa oculta
        """        
        super(NN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(n_input, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.out = nn.Linear(fc2_units, n_output)
    
    def forward(self, input):
        """Calcular salida de la red. No se debe invocar directamente
        Params
        ======
            input (): """
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        return self.out(x)