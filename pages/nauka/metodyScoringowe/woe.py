import streamlit as st
import pandas as pd
import numpy as np

st.info('''WOE jest metodą transformacji zmiennych, która przekształca zmienne kategorialne i numeryczne na wartości, które lepiej oddzielają klasy w modelu logistycznym. Formuła WOE to logarytm ilorazu procentu dobrych i złych klientów w danej kategorii.
        \n- WOE przekształca zmienne kategorialne na wartości numeryczne, które lepiej oddzielają klasy.
        \n- WOE jest używane w modelach predykcyjnych, aby poprawić separację klas i zmniejszyć wpływ outlierów.
        \n- WOE ułatwia interpretację wpływu poszczególnych kategorii na wynik modelu.
        \n- WOE jest szczególnie przydatne w regresji logistycznej, ponieważ przekształca zmienne na wartości liniowo związane z logitami prawdopodobieństwa, co poprawia wydajność modelu.''')

with st.expander('1.Dane wejściowe'):
    # Przykładowe dane
    data = {'age_category': ['<25', '<25', '<25', '25-35', '25-35', '25-35', '35-45', '35-45', '35-45', '>45', '>45', '>45'],
            'status': ['good', 'bad', 'bad', 'good', 'good', 'bad', 'good', 'bad', 'bad', 'good', 'good', 'bad']}

    df = pd.DataFrame(data)
    st.dataframe(data)

with st.expander('2.Kodowanie statusu (1 = good, 0 = bad)'):
    # Kodowanie statusu (1 = good, 0 = bad)
    df['status'] = df['status'].map({'good': 1, 'bad': 0})

    st.write(df)

with st.expander('3.Grupowanie danych według kategorii wiekowej i obliczanie liczby dobrych i złych klientów'):
    # Grupowanie danych według kategorii wiekowej i obliczanie liczby dobrych i złych klientów
    grouped = df.groupby('age_category')['status'].agg(['count', 'sum'])
    grouped.columns = ['total', 'good']
    grouped['bad'] = grouped['total'] - grouped['good']
    st.write(grouped)

with st.expander('4.Obliczanie procentu dobrych i złych klientów i WOE'):
    st.info('WOE: ' + r"$\ln \left( \frac{\% \text{good}}{\% \text{bad}} \right)$")
    # Obliczanie procentu dobrych i złych klientów
    grouped['%good'] = grouped['good'] / grouped['good'].sum()
    grouped['%bad'] = grouped['bad'] / grouped['bad'].sum()

    # Obliczanie WOE
    grouped['WOE'] = np.log((grouped['%good']+0.5) / (grouped['%bad']+0.5))

    st.write(grouped[['good', 'bad', '%good', '%bad', 'WOE']])

with st.expander('5.Dodanie kolumny WOE do oryginalnego DataFrame.'):
    # Dodanie kolumny WOE do oryginalnego DataFrame
    woe_dict = grouped['WOE'].to_dict()
    df['WOE'] = df['age_category'].map(woe_dict)

    st.write(df)