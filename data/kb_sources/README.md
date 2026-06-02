# Fontes das Bases de Conhecimento

Esta pasta guarda os documentos-fonte das KBs (FAQ bancário, políticas de
concessão, políticas de renegociação, catálogo de produtos).

Em produção, cada documento é ingerido pelo pipeline versionado e indexado no
vector DB. Cada chunk recebe os metadados obrigatórios: `versao_politica`,
`data_vigencia`, `produto`, `segmento`. Só políticas **vigentes na data atual**
são recuperadas.

Neste repositório de referência, as KBs são populadas em código por
`src/credito_agentes/kb/registry.py` (exemplos didáticos e determinísticos),
para que o sistema rode 100% offline.
