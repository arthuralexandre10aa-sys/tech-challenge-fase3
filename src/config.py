"""Configurações centrais do projeto (caminhos, semente, mapeamentos)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_EXT = ROOT / "data" / "external"
DATA_MANUAL = DATA_EXT / "manual"
DATA_PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
IMAGES = ROOT / "images"

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Transição usada para treinar: desempenho em ANO_ANT -> resultado em ANO_ALVO
ANO_ANT, ANO_ALVO = 2023, 2024

TARGET = "risco_nao_atingir_meta"  # 1 = município NÃO atingiu a meta (classe de risco)

# Colunas que definem o alvo ou são medidas depois dele: NUNCA podem ser features.
FORBIDDEN_FEATURES = {
    "pct_alvo", "prof_alvo", "gap_para_meta", "atingiu_meta", TARGET,
    "percentual_alfabetizado", "proficiencia_media",
}

ID_COLS = ["id_municipio", "nome_municipio"]

UF_REGIAO = {
    **dict.fromkeys(["AC", "AM", "AP", "PA", "RO", "RR", "TO"], "Norte"),
    **dict.fromkeys(["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"], "Nordeste"),
    **dict.fromkeys(["DF", "GO", "MT", "MS"], "Centro-Oeste"),
    **dict.fromkeys(["ES", "MG", "RJ", "SP"], "Sudeste"),
    **dict.fromkeys(["PR", "RS", "SC"], "Sul"),
}
