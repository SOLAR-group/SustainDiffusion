import pandas as pd

def pareto_front_full(df):
    df = df.sort_values(by=['Image Quality', 'Gender Fitness', 'Ethnicity Fitness', 'Duration', 'CPU Energy', 'GPU Energy'], ascending=[False, True, True, True, True, True])
    pareto = []
    for i in range(len(df)):
        if not any((df.iloc[i]['Image Quality'] <= p['Image Quality'] and
                    df.iloc[i]['Duration'] >= p['Duration'] and
                    df.iloc[i]['CPU Energy'] >= p['CPU Energy'] and
                    df.iloc[i]['GPU Energy'] >= p['GPU Energy'] and
                    df.iloc[i]['Gender Fitness'] >= p['Gender Fitness'] and
                    df.iloc[i]['Ethnicity Fitness'] >= p['Ethnicity Fitness']) for p in pareto):
            pareto.append(df.iloc[i])
    return pd.DataFrame(pareto)