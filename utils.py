import sys
from typing import List, Optional
from dados import InstanciaProblema, Solucao

def ler_instancia(caminho_arquivo: str) -> Optional[InstanciaProblema]:
    """
    Lê um arquivo de instância no formato especificado (com linhas em branco)
    e cria um objeto InstanciaProblema.

    Args:
        caminho_arquivo (str): O nome (e caminho) do arquivo a ser lido.

    Returns:
        Optional[InstanciaProblema]: Um objeto com os dados da instância,
                                     ou None se ocorrer um erro na leitura.
    """
    try:
        with open(caminho_arquivo, 'r') as f:
            # --- Leitura dos Cabeçalhos ---
            num_voos = int(f.readline().strip())
            num_pistas = int(f.readline().strip())

            # --- Pular a linha em branco APÓS num_pistas ---
            f.readline() # Lê e descarta a linha em branco

            # --- Leitura dos Arrays r, c, p ---
            tempos_liberacao = list(map(int, f.readline().strip().split()))
            tempos_processamento = list(map(int, f.readline().strip().split()))
            penalidades = list(map(int, f.readline().strip().split()))

            # --- Pular a linha em branco APÓS penalidades ---
            f.readline() # Lê e descarta a linha em branco

            # --- Leitura da Matriz t ---
            tempos_preparacao = []
            for _ in range(num_voos):
                linha = list(map(int, f.readline().strip().split()))
                tempos_preparacao.append(linha)

            # --- Validação básica ---
            # Verifica se o número de elementos lidos corresponde ao num_voos
            if len(tempos_liberacao) != num_voos:
                raise ValueError(f"Erro: Número incorreto de tempos de liberação. Esperado {num_voos}, lido {len(tempos_liberacao)}.")
            if len(tempos_processamento) != num_voos:
                 raise ValueError(f"Erro: Número incorreto de tempos de processamento. Esperado {num_voos}, lido {len(tempos_processamento)}.")
            if len(penalidades) != num_voos:
                 raise ValueError(f"Erro: Número incorreto de penalidades. Esperado {num_voos}, lido {len(penalidades)}.")
            # Verifica as dimensões da matriz t
            if len(tempos_preparacao) != num_voos:
                 raise ValueError(f"Erro: Número incorreto de linhas na matriz de preparação. Esperado {num_voos}, lido {len(tempos_preparacao)}.")
            for i, linha in enumerate(tempos_preparacao):
                if len(linha) != num_voos:
                     raise ValueError(f"Erro: Linha {i+1} da matriz de preparação t tem tamanho incorreto. Esperado {num_voos}, lido {len(linha)}.")

            # Se passou por todas as validações, cria e retorna o objeto
            return InstanciaProblema(num_voos, num_pistas, tempos_liberacao,
                                     tempos_processamento, penalidades, tempos_preparacao)

    except FileNotFoundError:
        print(f"Erro: Arquivo de instância não encontrado em '{caminho_arquivo}'", file=sys.stderr)
        return None
    except ValueError as ve:
        # Captura erros de conversão (int) e os erros levantados nas validações
        print(f"Erro de valor/formato ao ler o arquivo '{caminho_arquivo}': {ve}", file=sys.stderr)
        return None
    except Exception as e:
        # Captura outros erros inesperados (ex: permissão de leitura)
        print(f"Erro inesperado ao ler o arquivo '{caminho_arquivo}': {e}", file=sys.stderr)
        return None

def escrever_solucao(solucao: Solucao, caminho_arquivo: str):
    """
    Escreve a solução (penalidade total e escalonamento) em um arquivo,
    no formato pedido pelo PDF.

    Args:
        solucao (Solucao): O objeto Solucao contendo o escalonamento.
        caminho_arquivo (str): Nome do arquivo onde a solução será salva.
    """
    try:
        # Garante que a penalidade está calculada/atualizada
        valor_a_escrever = solucao.calcular_penalidade_total()

        with open(caminho_arquivo, 'w') as f:
            # 1. Escreve o valor da penalidade total
            f.write(f"{valor_a_escrever}\n")

            # 2. Escreve a sequência de voos para cada pista
            for pista in solucao.escalonamento:
                # Mapeia índices (0 a N-1) para códigos (1 a N)
                voos_str = " ".join(map(str, [idx + 1 for idx in pista]))
                f.write(voos_str + "\n")
    except IOError as e:
         print(f"Erro ao escrever o arquivo de solução '{caminho_arquivo}': {e}", file=sys.stderr)
    except Exception as e:
         print(f"Erro inesperado ao escrever a solução: {e}", file=sys.stderr)