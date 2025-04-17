import sys
import time
import argparse
from dados import InstanciaProblema, Solucao
from utils import ler_instancia, escrever_solucao


#pegar o voo disponivel com menor tempo de liberação
def algoritmo_guloso(instancia: InstanciaProblema) -> Solucao:
    solucao = Solucao(instancia)
    num_voos = instancia.num_voos
    num_pistas = instancia.num_pistas

    voos_agendados = [False] * num_voos
    voos_restantes = num_voos

    # Tempo em que a pista termina sua última tarefa agendada
    tempo_termino_pista = [0.0] * num_pistas
    # Índice do último voo agendado em cada pista (-1 se vazia)
    ultimo_voo_na_pista = [-1] * num_pistas

    #melhor voo
    while voos_restantes > 0:
        menor_tempo_liberacao = float('inf')
        voo_menor_tempo_liberacao = -1

        for i in range(num_voos):
            if instancia.tempos_liberacao[i] < menor_tempo_liberacao and voos_agendados[i] == False:
                voo_menor_tempo_liberacao = i
                menor_tempo_liberacao = instancia.tempos_liberacao[i]
        
        # A melhor pista é aquela onde o voo pode começar mais cedo.
        melhor_pista_idx = -1              # Índice da melhor pista encontrada
        melhor_horario_inicio = float('inf') # Menor horário de início encontrado
        tempo_termino_calculado = float('inf') # Tempo de término correspondente

        #melhor pista
        for p_idx in range(num_pistas):
            # Pega o índice do último voo que usou esta pista
            ultimo_voo = ultimo_voo_na_pista[p_idx]

            # Calcula o tempo de preparação necessário (t_ij)
            tempo_prep = 0.0
            if ultimo_voo != -1: # Se a pista não estiver vazia
                tempo_prep = instancia.tempos_preparacao[ultimo_voo][voo_menor_tempo_liberacao]

            # Calcula o tempo em que a PISTA estaria livre + preparação
            inicio_possivel_pista = tempo_termino_pista[p_idx] + tempo_prep
            # Pega o tempo em que o VOO está liberado (r_i)
            liberacao_voo = instancia.tempos_liberacao[voo_menor_tempo_liberacao]

            # O voo só pode começar quando AMBOS estiverem prontos
            horario_inicio_nesta_pista = max(inicio_possivel_pista, liberacao_voo)

            # Verifica se este horário de início é melhor que o melhor encontrado até agora
            if horario_inicio_nesta_pista < melhor_horario_inicio:
                melhor_horario_inicio = horario_inicio_nesta_pista
                melhor_pista_idx = p_idx
                # Calcula qual seria o tempo de término nesta pista
                tempo_termino_calculado = melhor_horario_inicio + instancia.tempos_processamento[voo_menor_tempo_liberacao]
            # Critério de desempate implícito: em caso de horários de início iguais,
            # a pista com menor índice 'p_idx' será escolhida.
            
        if melhor_pista_idx != -1:
            # Adiciona o voo ao final da lista da pista escolhida
            solucao.escalonamento[melhor_pista_idx].append(voo_menor_tempo_liberacao)
            # Marca o voo como agendado
            voos_agendados[voo_menor_tempo_liberacao] = True
            voos_restantes -= 1
            # Atualiza o estado da pista que recebeu o voo
            tempo_termino_pista[melhor_pista_idx] = tempo_termino_calculado
            ultimo_voo_na_pista[melhor_pista_idx] = voo_menor_tempo_liberacao
        else:
            # Se, por algum motivo, não encontrar pista (não deveria acontecer)
            print(f"ERRO: Não foi possível encontrar uma pista adequada para o voo {voo_menor_tempo_liberacao+1}", file=sys.stderr)
            break # Sai do loop
    return solucao


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resolve o Problema de Escalonamento de Voos.")
    parser.add_argument("arquivo_instancia", help="Caminho para o arquivo de instância do problema.")
    parser.add_argument("-o", "--arquivo_saida", default="solucao.txt", help="Caminho para salvar o arquivo com a solução final.")

    args = parser.parse_args()

    instancia = ler_instancia(args.arquivo_instancia)

    tempo_inicio = time.perf_counter()
    solucao_final = algoritmo_guloso(instancia)
    tempo_fim = time.perf_counter()
    tempo_execucao = (tempo_fim - tempo_inicio)

    print(f"Tempo de Execução: {tempo_execucao:.4f} segundos")

    escrever_solucao(solucao_final, args.arquivo_saida)