from typing import List

class InstanciaProblema:
    """
    Guarda todos os dados lidos do arquivo de instância do problema.
    Foca nos dados necessários para o cálculo de penalidade.
    """
    def __init__(self,
                 num_voos: int,
                 num_pistas: int,
                 tempos_liberacao: List[int],    # Corresponde ao array 'r' no PDF
                 tempos_processamento: List[int], # Corresponde ao array 'c' no PDF
                 penalidades: List[int],        # Corresponde ao array 'p' no PDF
                 tempos_preparacao: List[List[int]] # Corresponde à matriz 't' no PDF
                ):
        """
        Inicializa a instância do problema com os dados lidos.

        Args:
            num_voos (int): O número total de voos a serem escalonados (N).
            num_pistas (int): O número de pistas disponíveis no aeroporto (m).
            tempos_liberacao (List[int]): Lista r_i (tempo mínimo de início).
            tempos_processamento (List[int]): Lista c_i (tempo de ocupação da pista).
                                              Nota: Embora 'c' não esteja na fórmula da
                                              penalidade, é necessário para saber *quando*
                                              a pista fica livre para o próximo voo.
            penalidades (List[int]): Lista p_i (penalidade por unidade de atraso).
            tempos_preparacao (List[List[int]]): Matriz t_ij (tempo de segurança entre voos).
        """
        self.num_voos = num_voos
        self.num_pistas = num_pistas
        self.tempos_liberacao = tempos_liberacao
        self.tempos_processamento = tempos_processamento
        self.penalidades = penalidades
        self.tempos_preparacao = tempos_preparacao

# ----- CLASSE PARA REPRESENTAR UMA SOLUÇÃO (ESCALONAMENTO) -----
class Solucao:
    """
    Representa uma solução (escalonamento) para o problema, focada em
    minimizar a penalidade total.
    """
    def __init__(self, instancia: InstanciaProblema):
        """
        Inicializa um objeto Solucao vazio.

        Args:
            instancia (InstanciaProblema): A instância do problema associada.
        """
        self.instancia = instancia # Referência para os dados do problema

        # O escalonamento: lista de listas. Cada lista interna é uma pista
        # com os ÍNDICES (0 a N-1) dos voos na ordem de execução.
        self.escalonamento: List[List[int]] = [[] for _ in range(instancia.num_pistas)]

        # Penalidade Total: A soma das penalidades de todos os voos (Objetivo).
        self.penalidade_total: float = 0.0 # Inicializa com 0

    def calcular_penalidade_total(self) -> float:
        """
        Calcula a penalidade total desta solução, conforme o objetivo do PDF.
        Atualiza o atributo self.penalidade_total e retorna o valor.

        Returns:
            float: O valor da penalidade total calculado.
        """
        penalidade_acumulada = 0.0
        tempos_termino_pista = [0.0] * self.instancia.num_pistas # Track completion time for each runway

        # Itera sobre cada pista para calcular os tempos e penalidades
        for indice_pista in range(self.instancia.num_pistas):
            tempo_atual_pista = 0.0 # Tempo em que a pista fica livre
            ultimo_voo_idx = -1     # Índice do último voo na pista
            pista = self.escalonamento[indice_pista]

            # Itera sobre os voos na sequência definida para esta pista
            for voo_idx in pista:
                tempo_prep = 0.0
                # Calcula tempo de preparação se não for o primeiro voo
                if ultimo_voo_idx != -1:
                    tempo_prep = self.instancia.tempos_preparacao[ultimo_voo_idx][voo_idx]

                # Tempo em que a pista estaria livre após o voo anterior + preparação
                tempo_inicio_possivel_pista = tempo_atual_pista + tempo_prep
                # Tempo em que o voo está pronto para iniciar (r_i)
                tempo_liberacao_voo = self.instancia.tempos_liberacao[voo_idx]

                # O voo só pode começar quando AMBOS estiverem satisfeitos
                tempo_inicio_real = max(tempo_inicio_possivel_pista, tempo_liberacao_voo)

                # Calcula o atraso (somente se positivo)
                atraso = max(0.0, tempo_inicio_real - tempo_liberacao_voo)

                # Calcula a penalidade deste voo e acumula
                penalidade_voo = self.instancia.penalidades[voo_idx] * atraso
                penalidade_acumulada += penalidade_voo

                # Atualiza o tempo em que ESTA PISTA ficará livre
                # (Início real + tempo que o voo ocupa a pista)
                tempo_atual_pista = tempo_inicio_real + self.instancia.tempos_processamento[voo_idx]

                # Atualiza o último voo processado nesta pista
                ultimo_voo_idx = voo_idx

            # Guarda o tempo de término desta pista (informativo, não é o makespan)
            # tempos_termino_pista[indice_pista] = tempo_atual_pista

        self.penalidade_total = penalidade_acumulada
        return self.penalidade_total

    def copiar(self) -> 'Solucao':
        """
        Cria uma cópia INDEPENDENTE desta solução.
        Útil para busca local.

        Returns:
            Solucao: Uma nova instância de Solucao com os mesmos dados.
        """
        nova_sol = Solucao(self.instancia)
        # Copia profunda do escalonamento
        nova_sol.escalonamento = [list(pista) for pista in self.escalonamento]
        # Copia o valor da penalidade (pode ser recalculado se preferir)
        nova_sol.penalidade_total = self.penalidade_total
        return nova_sol