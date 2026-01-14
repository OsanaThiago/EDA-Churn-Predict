class Recomendar:
    def recomendacao(self, probachurn, shap_df):
        recomendacoes = []

        if probachurn > 0.65:
            recomendacoes.extend(["Contato imediato", "Oferta personalizada", "Revisão de tarifas / benefícios"])

        elif probachurn > 0.30:
            recomendacoes.extend(["Campanha de engajamento", "Oferta de upgrade ou pontos bônus"])

        else:
            recomendacoes.extend(["Manter relacionamento", "Cross-sell de produtos"])

        shap_risco = shap_df[shap_df["Impacto"] > 0]["Feature"].values

        if "Satisfaction Score" in shap_risco:
            recomendacoes.append("Realizar ação de recuperação de satisfação (contato ativo, benefícios personalizados ou pesquisa NPS).")

        if "IsActiveMember" in shap_risco:
            recomendacoes.append("Incentivar maior engajamento com produtos e serviços por meio de campanhas direcionadas.")

        if "Balance" in shap_risco:
            recomendacoes.append("Revisar tarifas, condições financeiras ou oferecer vantagens para retenção do saldo.")

        if "NumOfProducts" in shap_risco:
            recomendacoes.append("Oferecer produtos complementares para aumentar o vínculo do cliente com a instituição.")

        if "Tenure" in shap_risco:
            recomendacoes.append("Reforçar ações de onboarding e acompanhamento nos primeiros anos de relacionamento.")

        if "Age" in shap_risco:
            recomendacoes.append("Adaptar abordagem de atendimento conforme o perfil etário do cliente.")

        if "CreditScore" in shap_risco:
            recomendacoes.append("Oferecer orientação financeira ou produtos adequados ao perfil de risco do cliente.")

        if "HasCrCard" in shap_risco:
            recomendacoes.append("Avaliar oferta de cartão de crédito com benefícios alinhados ao perfil do cliente.")

        if "Card Type" in shap_risco:
            recomendacoes.append("Avaliar upgrade de cartão ou inclusão de benefícios exclusivos.")

        if "Point Earned" in shap_risco:
            recomendacoes.append("Criar campanha de incentivo com acúmulo ou resgate facilitado de pontos.")

        if "Geography" in shap_risco:
            recomendacoes.append("Aplicar estratégia regional específica para reduzir churn nesse país.")

        if not recomendacoes:
            recomendacoes.append("Manter estratégia atual e monitorar o comportamento do cliente.")

        return recomendacoes
