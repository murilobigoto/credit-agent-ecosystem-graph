#!/usr/bin/env python3
"""
Validação de prompts (usada no CI).
===================================

DEFINIÇÃO — "validação de prompt": uma checagem automática de que os prompts
críticos do sistema (1) existem e (2) contêm as instruções obrigatórias. Como os
prompts moram no código, eles entram no versionamento e no code review; este
script é a rede de segurança que impede um prompt quebrado de chegar à produção.

Sai com código 0 se tudo passar; 1 caso contrário (falha o pipeline).
"""

from __future__ import annotations

import sys

from credito_agentes.orchestrator.router import _SYSTEM_ROTEADOR

# Cada regra: (nome legível, texto do prompt, lista de trechos obrigatórios).
CHECKS: list[tuple[str, str, list[str]]] = [
    (
        "roteador",
        _SYSTEM_ROTEADOR,
        ["JSON", "concessao", "renegociacao", "bancario_geral"],
    ),
]


def main() -> int:
    falhas: list[str] = []
    for nome, prompt, obrigatorios in CHECKS:
        if not prompt or not prompt.strip():
            falhas.append(f"[{nome}] prompt vazio")
            continue
        for termo in obrigatorios:
            if termo.lower() not in prompt.lower():
                falhas.append(f"[{nome}] falta o termo obrigatório: {termo!r}")

    if falhas:
        print("Validação de prompts FALHOU:")
        for f in falhas:
            print(f"  - {f}")
        return 1

    print(f"Validação de prompts OK ({len(CHECKS)} prompt(s) verificado(s)).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
