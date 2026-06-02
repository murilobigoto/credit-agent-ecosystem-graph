"""
Registro de bases de conhecimento por dominio.

DOMINIOS
--------
Quatro bases independentes evitam que termos de um dominio poluam a
recuperacao de outro (problema comum em bases unificadas).

  faq_bancario          -- perguntas frequentes do agente de fallback.
  politicas_concessao   -- elegibilidade, compliance e limites para concessao.
  politicas_renegociacao -- descontos, prazos e condicoes de renegociacao.
  catalogo_produtos     -- descricao e parametros dos produtos disponiveis.

INGESTAO VERSIONADA
-------------------
Cada documento entra com versao de politica e datas de vigencia. Isso permite:
  - filtro por vigencia na recuperacao (nunca citar norma revogada);
  - pipeline de aprovacao (so conteudo aprovado vai a producao);
  - rollback (voltar a versao anterior reinstanciando a KB).

Os textos abaixo refletem o formato real de FAQs e manuais de credito de
bancos de varejo brasileiros, incluindo referencias normativas corretas
(Resolucoes CMN, regulamentacao Bacen).
"""

from __future__ import annotations

from datetime import date

from credito_agentes.kb.retrieval import ChunkMetadata, KnowledgeBase


def _meta(versao: str, produto: str, segmento: str = "todos") -> ChunkMetadata:
    return ChunkMetadata(
        versao_politica=versao,
        vigencia_inicio=date(2024, 1, 1),
        vigencia_fim=None,
        produto=produto,
        segmento=segmento,
    )


def build_default_kbs() -> dict[str, KnowledgeBase]:
    """Constroi e popula as quatro KBs com conteudo realista e versionado."""

    # ------------------------------------------------------------------
    # FAQ BANCARIO
    # Conteudo tipico de FAQ de um banco de varejo digital brasileiro.
    # ------------------------------------------------------------------
    faq = KnowledgeBase("faq_bancario")

    faq.add_document(
        "Como consultar o saldo da conta corrente?\n"
        "Voce pode consultar seu saldo pelo aplicativo do Banco Poneiu na tela "
        "inicial, pelo internet banking em poneiu.com.br, nos terminais de "
        "autoatendimento, ou perguntando diretamente aqui no assistente virtual.\n\n"
        "Como ver o extrato da conta?\n"
        "O extrato completo esta disponivel no app em Conta > Extrato, com filtros "
        "por periodo de ate 90 dias. Para periodos maiores, acesse o internet "
        "banking ou va a uma agencia.\n\n"
        "Como solicitar o extrato anual para o imposto de renda?\n"
        "O informe de rendimentos para declaracao do IR fica disponivel a partir "
        "de fevereiro de cada ano no app e no internet banking, em Conta > "
        "Servicos > Informe de Rendimentos.",
        _meta("FAQ-v3", "geral"),
        prefixo="faq1",
    )

    faq.add_document(
        "Como funciona o PIX?\n"
        "PIX e o sistema de pagamentos instantaneos do Banco Central. Transferencias "
        "e pagamentos sao liquidados em ate 10 segundos, 24 horas por dia, 7 dias "
        "por semana, inclusive feriados. Nao ha custo para pessoas fisicas.\n\n"
        "Como cadastrar uma chave PIX?\n"
        "Acesse o app em Conta > PIX > Minhas Chaves > Cadastrar Nova Chave. "
        "Voce pode usar CPF, numero de telefone, e-mail ou chave aleatoria. "
        "Cada CPF pode ter ate 5 chaves cadastradas.\n\n"
        "Qual o limite do PIX?\n"
        "O limite padrao para transacoes PIX e de R$ 1.000,00 por transacao "
        "durante o dia e R$ 500,00 entre 20h e 6h (limite noturno). Voce pode "
        "solicitar alteracao do limite no app em Conta > PIX > Limites.",
        _meta("FAQ-v3", "geral"),
        prefixo="faq2",
    )

    faq.add_document(
        "Como ver a fatura do cartao de credito?\n"
        "A fatura fica disponivel no app em Cartoes > sua bandeira > Fatura Atual, "
        "com data de vencimento e valor minimo em destaque. Faturas anteriores "
        "ficam em Historico de Faturas.\n\n"
        "Como funciona o parcelamento da fatura?\n"
        "Voce pode parcelar o saldo da fatura em ate 12 vezes pelo app. A taxa "
        "de juros do parcelamento e informada antes da confirmacao. O parcelamento "
        "nao exclui o pagamento do valor minimo do mes vigente.\n\n"
        "Qual o vencimento do meu cartao?\n"
        "A data de vencimento esta no app em Cartoes > sua bandeira > Detalhes. "
        "Voce pode solicitar alteracao da data de vencimento uma vez por ano.",
        _meta("FAQ-v3", "cartao"),
        prefixo="faq3",
    )

    faq.add_document(
        "O que e o cheque especial?\n"
        "O cheque especial e uma linha de credito pre-aprovada vinculada a sua "
        "conta corrente. Quando o saldo fica negativo, o banco libera automaticamente "
        "esse limite. A taxa maxima e de 8% ao mes, conforme regulamentacao do "
        "Banco Central (Resolucao CMN 4.765/2019).\n\n"
        "Como funciona a cobranca do cheque especial?\n"
        "Juros sao cobrados diariamente sobre o saldo negativo utilizado. "
        "A cobranca aparece na fatura do mes seguinte. Uso acima de 5% do limite "
        "por mais de 30 dias consecutivos aciona oferta de portabilidade de divida "
        "em condicoes melhores.\n\n"
        "Como desativar o cheque especial?\n"
        "Voce pode desativar o cheque especial pelo app em Conta > Cheque Especial "
        "> Desativar. A desativacao e imediata e irreversivel ate nova solicitacao.",
        _meta("FAQ-v3", "cheque_especial"),
        prefixo="faq4",
    )

    faq.add_document(
        "Como bloquear o cartao de credito em caso de perda ou roubo?\n"
        "Bloqueie imediatamente pelo app em Cartoes > sua bandeira > Bloquear "
        "Cartao, ou ligue 0800 722 0001 (24h). O bloqueio e instantaneo e "
        "impede novas compras. Para solicitar o segundo via, acesse Cartoes > "
        "Solicitar Segunda Via apos o bloqueio.\n\n"
        "Como contestar uma compra no cartao?\n"
        "Em caso de compra nao reconhecida, acesse o app em Cartoes > Fatura > "
        "selecione a transacao > Contestar. O prazo para contestacao e de 60 dias "
        "a partir da data da transacao. O valor e estornado provisoriamente em "
        "ate 2 dias uteis enquanto o caso e analisado.",
        _meta("FAQ-v3", "cartao"),
        prefixo="faq5",
    )

    faq.add_document(
        "Como fazer uma transferencia TED ou DOC?\n"
        "No app, acesse Transferir > TED/DOC, informe os dados bancarios do "
        "destinatario e confirme com sua senha. TED e liquidado no mesmo dia "
        "se enviado ate 17h em dias uteis. DOC e liquidado no proximo dia util.\n\n"
        "Qual o limite de transferencia?\n"
        "O limite padrao para TED/DOC e de R$ 5.000,00 por dia. Para aumentar "
        "o limite, acesse o app em Perfil > Limites e Seguranca > Transferencias. "
        "Aumentos acima de R$ 20.000,00 requerem verificacao presencial.\n\n"
        "Como proteger minha conta de fraudes?\n"
        "Nunca compartilhe sua senha, token ou codigo de verificacao com ninguem, "
        "nem por telefone, e-mail ou mensagem. O Banco Poneiu nunca solicita esses "
        "dados. Em caso de suspeita de fraude, bloqueie seu acesso digital no app "
        "em Perfil > Seguranca > Bloquear Acesso.",
        _meta("FAQ-v3", "geral"),
        prefixo="faq6",
    )

    # ------------------------------------------------------------------
    # POLITICAS DE CONCESSAO
    # Baseado no formato real de manuais de credito de bancos brasileiros
    # com referencias normativas corretas.
    # ------------------------------------------------------------------
    concessao = KnowledgeBase("politicas_concessao")

    concessao.add_document(
        "Politica de Elegibilidade - Credito Pessoal (POL-CONC-v4)\n"
        "Vigencia: 01/01/2024 a presente data.\n\n"
        "CRITERIOS MINIMOS PARA CONCESSAO:\n"
        "1. Cadastro completo e atualizado (RG, CPF, comprovante de renda e "
        "residencia atualizados ha menos de 12 meses).\n"
        "2. Ausencia de restricoes ativas nos birôs de credito (Serasa, SCR/Bacen).\n"
        "3. Score bureau minimo de 500 pontos (escala 0-1000).\n"
        "4. Relacionamento minimo de 3 meses com o banco.\n"
        "5. Renda mensal comprovada de pelo menos R$ 1.500,00.\n\n"
        "COMPROMETIMENTO DE RENDA:\n"
        "O total de parcelas de credito (incluindo a nova operacao) nao pode "
        "exceder 35% da renda mensal liquida do cliente, conforme Resolucao "
        "CMN 4.966/2021 e politica interna de credito.",
        _meta("POL-CONC-v4", "credito_pessoal"),
        prefixo="conc1",
    )

    concessao.add_document(
        "Limites Regulatorios e Parametros de Compliance - Credito Pessoal\n\n"
        "TAXA DE JUROS:\n"
        "Taxa mensal minima: 1,20% a.m. (conforme teto regulatorio CMN).\n"
        "Taxa base por segmento:\n"
        "  - Varejo: 1,89% a.m.\n"
        "  - Alta Renda: 1,49% a.m.\n"
        "  - PJ (pequena empresa): 1,69% a.m.\n\n"
        "PRAZO:\n"
        "Prazo maximo para credito pessoal sem garantia: 48 meses.\n"
        "Prazo maximo para credito com garantia (home equity, veiculo): 120 meses.\n\n"
        "LIMITE:\n"
        "Limite maximo para varejo: R$ 30.000,00.\n"
        "Limite maximo para alta renda: R$ 100.000,00.\n"
        "Limite calculado com base na capacidade de pagamento apurada pelo modelo "
        "de scoring interno, respeitando o comprometimento maximo de 35% da renda.\n\n"
        "CET (Custo Efetivo Total): deve ser informado antes da contratacao, "
        "conforme Resolucao CMN 3.517/2007.",
        _meta("POL-CONC-v4", "credito_pessoal"),
        prefixo="conc2",
    )

    concessao.add_document(
        "Politica de Escalada para Analise Humana - Credito Pessoal\n\n"
        "CASOS QUE REQUEREM ANALISE MANUAL:\n"
        "1. Score de inadimplencia entre 0,45 e 0,55 (zona limitrofe).\n"
        "2. Renda declarada superior a R$ 50.000,00/mes (cliente VIP).\n"
        "3. Solicitacao de limite acima de R$ 50.000,00.\n"
        "4. Cliente com historico de renegociacao nos ultimos 24 meses.\n"
        "5. Operacoes com suspeita de fraude flagradas pelo modelo antifraude.\n\n"
        "CRITERIOS DE APROVACAO AUTOMATICA:\n"
        "Score de inadimplencia <= 0,44 E politicas favoraveis E "
        "comprometimento <= 35% -> aprovacao automatica.\n"
        "Score de inadimplencia >= 0,56 -> recusa automatica.\n\n"
        "PRAZO DE ANALISE HUMANA: ate 1 dia util para varejo, ate 3 dias uteis "
        "para casos complexos.",
        _meta("POL-CONC-v4", "credito_pessoal"),
        prefixo="conc3",
    )

    # ------------------------------------------------------------------
    # POLITICAS DE RENEGOCIACAO
    # ------------------------------------------------------------------
    reneg = KnowledgeBase("politicas_renegociacao")

    reneg.add_document(
        "Politica de Renegociacao de Dividas (POL-RENEG-v3)\n"
        "Vigencia: 01/01/2024 a presente data.\n\n"
        "CONDICOES DE ACESSO:\n"
        "Cliente elegivel para renegociacao quando:\n"
        "  - Possui parcelas em atraso ha mais de 30 dias, OU\n"
        "  - Saldo devedor total superior a R$ 500,00.\n\n"
        "OPCOES DE PARCELAMENTO:\n"
        "  - Pagamento a vista: desconto de ate 40% sobre o saldo devedor.\n"
        "  - Parcelamento em 6 meses: desconto de ate 30%.\n"
        "  - Parcelamento em 12 meses: desconto de ate 25%.\n"
        "  - Parcelamento em 24 meses: desconto de ate 15%.\n"
        "  - Parcelamento em 48 meses: desconto de ate 10%.\n"
        "  - Parcelamento em 60 meses: sem desconto (excecao aprovada por gerente).\n\n"
        "As condicoes acima sao teto; o desconto efetivo e calculado pelo "
        "modelo de probabilidade de cura (prob_cura) do cliente.",
        _meta("POL-RENEG-v3", "recuperacao"),
        prefixo="reneg1",
    )

    reneg.add_document(
        "Taxas de Juros e Limites em Renegociacao (POL-RENEG-v3)\n\n"
        "TAXA MINIMA EM RENEGOCIACAO: 0,90% a.m.\n"
        "A taxa minima de 0,90% a.m. aplica-se independentemente do desconto, "
        "conforme resolucao interna de conformidade.\n\n"
        "TAXA POR PRAZO:\n"
        "  - 6 meses:  1,20% a.m.\n"
        "  - 12 meses: 1,22% a.m.\n"
        "  - 24 meses: 1,25% a.m.\n"
        "  - 48 meses: 1,30% a.m.\n"
        "  - 60 meses: 1,32% a.m.\n\n"
        "DESCONTO MAXIMO ABSOLUTO: 40% do saldo devedor original (sem juros).\n\n"
        "CLIENTES COM ALTA PROBABILIDADE DE CURA (>= 0,70):\n"
        "Recebem condicoes preferenciais: desconto maximo do prazo acrescido de "
        "5 pontos percentuais e isencao de multa moratoria.",
        _meta("POL-RENEG-v3", "recuperacao"),
        prefixo="reneg2",
    )

    reneg.add_document(
        "Processo de Renegociacao e Documentacao (POL-RENEG-v3)\n\n"
        "FLUXO DE RENEGOCIACAO:\n"
        "1. Cliente solicita renegociacao (assistente virtual, app ou agencia).\n"
        "2. Sistema consulta contratos ativos e calcula opcoes automaticamente.\n"
        "3. Cliente escolhe a opcao que melhor se adequa.\n"
        "4. Confirmacao com autenticacao forte (biometria ou token).\n"
        "5. Emissao de boleto ou debito automatico da primeira parcela.\n"
        "6. Certificado de renegociacao disponivel no app em ate 24h.\n\n"
        "DOCUMENTACAO NECESSARIA:\n"
        "  - Documentos de identificacao (RG/CNH).\n"
        "  - Comprovante de renda atualizado (ultimos 3 meses).\n"
        "  - Comprovante de residencia (ultimos 3 meses).\n\n"
        "PRAZO PARA PROTOCOLO: a renegociacao deve ser protocolada ate o vencimento "
        "do contrato mais antigo em atraso. Apos esse prazo, o caso e encaminhado "
        "para cobranca judicial.",
        _meta("POL-RENEG-v3", "recuperacao"),
        prefixo="reneg3",
    )

    # ------------------------------------------------------------------
    # CATALOGO DE PRODUTOS
    # ------------------------------------------------------------------
    catalogo = KnowledgeBase("catalogo_produtos")

    catalogo.add_document(
        "Credito Pessoal Facil - Banco Poneiu\n\n"
        "DESCRICAO:\n"
        "Credito sem garantia para qualquer finalidade: reforma, viagem, estudos, "
        "emergencias medicas ou o que o cliente preferir.\n\n"
        "PARAMETROS:\n"
        "  - Limite: de R$ 500,00 a R$ 30.000,00 (varejo) / R$ 100.000,00 (alta renda).\n"
        "  - Taxa: a partir de 1,49% a.m. (alta renda) ou 1,89% a.m. (varejo).\n"
        "  - Prazo: de 6 a 48 meses.\n"
        "  - Liberacao: em ate 10 minutos apos aprovacao, diretamente na conta.\n\n"
        "REQUISITOS:\n"
        "  - Score bureau >= 500.\n"
        "  - Relacionamento minimo de 3 meses.\n"
        "  - Renda mensal >= R$ 1.500,00.\n\n"
        "CET REFERENCIAL: 2,1% a.m. (varejo, prazo de 24 meses).",
        _meta("CAT-v6", "credito_pessoal"),
        prefixo="cat1",
    )

    catalogo.add_document(
        "Credito Consignado Poneiu\n\n"
        "DESCRICAO:\n"
        "Credito com desconto em folha de pagamento para servidores publicos, "
        "aposentados INSS e funcionarios de empresas conveniadas.\n\n"
        "PARAMETROS:\n"
        "  - Taxa: a partir de 1,20% a.m. (menor taxa da carteira).\n"
        "  - Prazo: ate 96 meses para servidores, ate 84 meses para INSS.\n"
        "  - Limite: ate 35% da renda liquida (margem consignavel).\n"
        "  - Liberacao: em ate 2 dias uteis apos assinatura digital.\n\n"
        "DIFERENCIAIS:\n"
        "  - Parcela fixa descontada automaticamente na folha/beneficio.\n"
        "  - Aprovacao simplificada (sem consulta ao bureau para conveniadas).\n"
        "  - Portabilidade: voce pode trazer seu consignado de outro banco.\n\n"
        "REQUISITOS:\n"
        "  - Vinculo empregaticio ou beneficio INSS ativo.\n"
        "  - Margem consignavel disponivel.\n"
        "  - Empresa ou orgao conveniado ao Banco Poneiu.",
        _meta("CAT-v6", "credito_consignado"),
        prefixo="cat2",
    )

    catalogo.add_document(
        "Cartao de Credito Poneiu Platinum\n\n"
        "DESCRICAO:\n"
        "Cartao de credito sem anuidade no primeiro ano, com programa de pontos "
        "Poneiu Rewards e beneficios de viagem.\n\n"
        "PARAMETROS:\n"
        "  - Limite: de R$ 1.000,00 a R$ 15.000,00 conforme analise.\n"
        "  - Taxa de juros rotativos: 14,9% a.m. (teto legal).\n"
        "  - Anuidade: R$ 0 no primeiro ano; R$ 360,00/ano a partir do segundo "
        "    (isenta com gasto minimo de R$ 800,00/mes).\n\n"
        "BENEFICIOS:\n"
        "  - 1,5 ponto por real gasto em compras nacionais.\n"
        "  - 3,0 pontos por real em compras internacionais.\n"
        "  - Seguro viagem e seguro compra inclusos.\n"
        "  - Acesso a salas VIP em aeroportos com 4 visitas/ano.\n\n"
        "COMO SOLICITAR:\n"
        "  - Clientes com conta ha mais de 6 meses podem solicitar pelo app.\n"
        "  - Novos clientes: solicitacao presencial ou via internet banking.",
        _meta("CAT-v6", "cartao"),
        prefixo="cat3",
    )

    catalogo.add_document(
        "Cheque Especial Poneiu\n\n"
        "DESCRICAO:\n"
        "Limite de credito pre-aprovado vinculado a conta corrente, disponivel "
        "automaticamente quando o saldo fica negativo.\n\n"
        "PARAMETROS:\n"
        "  - Limite: de R$ 200,00 a R$ 5.000,00 conforme perfil.\n"
        "  - Taxa maxima: 8% a.m. (teto fixado pelo Banco Central, "
        "    Resolucao CMN 4.765/2019).\n"
        "  - Cobranca diaria sobre o saldo negativo utilizado.\n\n"
        "ALTERNATIVA RECOMENDADA:\n"
        "O uso prolongado do cheque especial e mais caro do que um credito pessoal. "
        "Se o saldo ficar negativo por mais de 5 dias consecutivos, o app exibe "
        "automaticamente uma oferta de portabilidade da divida para credito pessoal "
        "a taxas menores.\n\n"
        "COMO AJUSTAR O LIMITE:\n"
        "  - Reducao: imediata pelo app.\n"
        "  - Aumento: sujeito a nova analise de credito.",
        _meta("CAT-v6", "cheque_especial"),
        prefixo="cat4",
    )

    return {
        "faq_bancario": faq,
        "politicas_concessao": concessao,
        "politicas_renegociacao": reneg,
        "catalogo_produtos": catalogo,
    }
