def aplicarRegla(df):
    """
    Aplica la regla creada de manera manual:
    si Humidity > 75 then no juega
    si Humidity <=75 then juega
    """
    df["Prediccion"] = df.apply(reglaDecision, axis=1)
    # conteo de la instancia con "Play"=='yes'
    aciertos = (df["Prediccion"] == df["Play"]).sum()
    # ponderación
    precision = aciertos / len(df)
    return df["Prediccion"], aciertos, precision


def reglaDecision(instancia):

    # Regla 1: Si lluvia y sin viento, juega
    if (instancia["Outlook"] == "rain") and (instancia["Windy"] == False):
        return "yes"

    # if instancia["Outlook"] == "overcast":
    #     return "yes"
    return "no"
