class Recomendar:
    def recomendacao(self, probachurn, shap_df):
        recomendacoes = []

        if probachurn > 0.65:
            recomendacoes.extend([ "Contato imediato", "Oferta personalizada", "Revisão de tarifas / benefícios"])

        elif probachurn > 0.30:
            recomendacoes.extend(["Campanha de engajamento", "Oferta de upgrade ou pontos bônus"])

        else:
            recomendacoes.extend(["Manter relacionamento", "Cross-sell de produtos"])

        if "Satisfaction Score" in shap_df["Feature"].values:
            recomendacoes.append("Avaliar satisfação do cliente e oferecer benefícios personalizados")

        if "IsActiveMember" in shap_df["Feature"].values:
            recomendacoes.append("Incentivar maior engajamento com produtos e serviços")

        if "Balance" in shap_df["Feature"].values:
            recomendacoes.append("Revisar condições financeiras e tarifas aplicadas")

        return recomendacoes
