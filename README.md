# Atividade 2 — Redes MLP com PyTorch
**Tópicos para Computação 1 — 2026.1**  
Escola Superior de Tecnologia — UEA  
Profa. Dra. Elloá B. Guedes

---

## Alunos

| Nome | Matrícula |
|---|---|
| Caio Jorge da Cunha Queiroz | 2315310028 |
| Lucas Maciel Gomes | 2315310014 |
| Izabella de Lima Catrinck | 2315310033 |

---

## Descrição

Classificação binária para prever se a renda anual de um adulto excede **US$ 50.000**, utilizando o [UCI Adult Income Dataset](https://archive.ics.uci.edu/dataset/2/adult). Foram propostas e comparadas três arquiteturas de redes neurais MLP em PyTorch, com análise exploratória dos dados, pré-processamento com Polars e avaliação com métricas balanceadas.

---

## Estrutura do projeto

```
.
├── adult/
│   ├── adult_train.csv      # Arquivo de treino do UCI Adult Dataset
│   └── adult_test.csv       # Arquivo de teste do UCI Adult Dataset
├── Topicos1-2026_1-Tarefa2.ipynb
└── README.md
```

---

## Dataset

O UCI Adult Income Dataset contém dados do censo americano de 1994, com 48.842 exemplos brutos e 15 atributos.

### Pré-processamento aplicado

**Filtros dos autores do dataset:**
```
age > 16  AND  capital-gain > 100  AND  fnlwgt > 1  AND  hours-per-week > 0
```
> O filtro `capital-gain > 100` é o mais impactante: descarta ~92% dos exemplos, selecionando apenas pessoas com ganhos de capital relevantes. Isso inverte a distribuição de classes em relação ao dataset original.

**Passos adicionais:**
- Remoção de linhas com dados faltantes (`?`) nas colunas `workclass`, `occupation` e `native-country`
- One-Hot Encoding das 8 colunas categóricas: `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`
- Normalização com `StandardScaler`
- Variável alvo: `<=50K → 0`, `>50K → 1`

**Resultado final:** ~3.790 exemplos, 101 atributos preditores

### Distribuição das classes (após filtro)

| Classe | Proporção |
|---|---|
| ≤ 50K | ~37% |
| > 50K | ~63% |

O dataset **não é balanceado**. A distribuição é invertida em relação ao original (~76% ≤50K) por efeito do filtro de `capital-gain`. Por isso, todas as métricas de avaliação usam a versão **macro/balanceada**.

---

## Partição dos dados

- Holdout **70% treino / 30% teste**
- `numpy.random.seed(42)` para reprodutibilidade
- Mini-batch SGD com `batch_size=16`

---

## Arquiteturas propostas

### MLP1 — Camada única
```
input (101) → Linear(100) → ReLU → Linear(1)
```
- Otimizador: SGD, `lr=10e-4`
- Épocas: 100
- Loss: `BCEWithLogitsLoss`

### MLP2 — Duas camadas ocultas (padrão)
```
input (101) → Linear(100) → ReLU → Linear(50) → ReLU → Linear(1)
```
- Otimizador: SGD, `lr=10e-4`
- Épocas: 100
- Loss: `BCEWithLogitsLoss`

### MLP3 — Nossa rede (duas camadas ocultas, mais larga)
```
input (101) → Linear(128) → ReLU → Linear(64) → ReLU → Linear(1)
```
- Otimizador: SGD com momentum, `lr=10e-4`, `momentum=0.9`
- Épocas: 150
- Loss: `BCEWithLogitsLoss`

> **Nota:** `BCEWithLogitsLoss` é usada no lugar de `BCELoss + Sigmoid` pois combina as duas operações de forma numericamente estável, evitando o erro `all elements of input should be between 0 and 1` que ocorre com precisão float32.

---

## Resultados

| Rede | Arquitetura | Épocas | lr | Acurácia Bal. | Precisão | Revocação | F1-Score |
|---|---|---|---|---|---|---|---|
| MLP1 | input→100→1 | 100 | 10e-4 | 0.51 | 0.53 | 0.51 | 0.35 |
| MLP2 | input→100→50→1 | 100 | 10e-4 | 0.52 | 0.52 | 0.52 | 0.48 |
| **MLP3** | **input→128→64→1** | **150** | **10e-4** | **0.85** | **0.84** | **0.85** | **0.85** |
| MLP4* | input→128→64→1 | 200 | 10e-4 | 0.84 | 0.84 | 0.84 | 0.84 |

*MLP4 é a mesma arquitetura da MLP3 com 200 épocas, usada para investigar o efeito de treino adicional.

Todas as métricas são calculadas com `average="macro"` (peso igual para cada classe), consistente com o uso de `balanced_accuracy_score`.

### Análise dos resultados

1. PERFORMANCE\
A Nossa rede MLP3 superou a MLP1 e MLP2.

2. EFICIÊNCIA\
MLP1 é a mais rápida de treinar (menos parâmetros e 100 épocas).\
MLP3, apesar de ter mais épocas (150), converge de forma mais suave.

3. ADERÊNCIA AO PROBLEMA\
O Adult Income Dataset é de classificação binária com features mistas (numéricas + OHE).\
A MLP3, com maior largura (128→64 vs 100→50) e otimização mais agressiva, conseguiu aprender representações úteis para ambas as classes, como evidenciam as métricas macro equilibradas (~0.85 em todas).

Conclusão: Nossa rede MLP3 obteve o melhor desempenho global nesta tarefa, consistente em todas as métricas balanceadas. O fator decisivo foi a escolha do otimizador: SGD com momentum é substancialmente mais eficaz que SGD puro para este dataset com mini-batch de tamanho 16.

---

## Dependências

```
polars
torch
numpy
matplotlib
scikit-learn
prettytable
```

## Como executar

### Com uv (recomendado)

```bash
uv sync
uv run jupyter notebook Topicos1-2026_1-Tarefa2.ipynb
```

### Com venv

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows

pip install torch polars matplotlib scikit-learn prettytable jupyter ipykernel
python -m ipykernel install --user --name=mlp-adult

jupyter notebook Topicos1-2026_1-Tarefa2.ipynb
```
> Ao abrir o notebook no Jupyter, selecione o kernel `mlp-adult` em vez do padrão do sistema.

**README feito com IA**