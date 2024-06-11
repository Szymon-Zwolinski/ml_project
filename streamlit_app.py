import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from data_files.data_lr_woe import data as data1
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import seaborn as sns


st.set_page_config(page_title="Main page", page_icon="🏠")

st.title("Metody Scoringowe i Techniki klasyfikacji danych")

st.divider()

ms, tkd = st.tabs(['Projekt MS', 'Techniki klasyfikacji danych'])

with ms: 
    # Załadowanie danych
    @st.cache_data
    def load_data(file):
        return pd.read_excel(file)

    data = pd.DataFrame(load_data('./data_files/ms_data.xlsx')[['Age_group', 'mth_salary_tsd_group', 'positive_credit_history', 'other_credits', 'target']])
    data.rename(columns={'Age_group':'age', 'mth_salary_tsd_group':'salary', 'positive_credit_history':'credit_hist'}, inplace=True)

    st.write('## Wstęp')
    st.write("""
    Niniejszy projekt prezentuje wyniki ewaluacji modelu ML do przewidywania przyznania kredytu (`target`). 
    Model został oceniony pod kątem dokładności (Accuracy) oraz obszaru pod krzywą ROC (ROC AUC).

    Wprowadzenie modeli scoringowych, takich jak regresja logistyczna, do procesu decyzyjnego przyznawania kredytów ma na celu zwiększenie precyzji i obiektywności oceny zdolności kredytowej klientów. Zastosowanie technik scoringowych pozwala na uwzględnienie wielu czynników wpływających na ryzyko kredytowe, co przekłada się na lepsze zarządzanie portfelem kredytowym instytucji finansowej.

    W ramach tego projektu wykorzystano dane zawierające informacje o klientach, takie jak grupa wiekowa, miesięczne wynagrodzenie, historia kredytowa oraz liczba posiadanych innych kredytów. Na podstawie tych danych model uczy się przewidywać, czy dany klient otrzyma kredyt (wartość 1) czy też nie (wartość 0). 

    W kolejnych sekcjach przedstawione są szczegóły dotyczące danych, obliczania wag dowodów (WOE), wartości informacyjnej (IV), a także wyniki ewaluacji modelu na różnych zbiorach danych. Prezentowane są również histogramy oraz macierz konfuzji, które pozwalają na wizualną ocenę rozkładu danych i skuteczności modelu.


    ## Opis danych
    Dane użyte do trenowania i ewaluacji modelu zawierają następujące kolumny:
    - `Age`: Grupa wiekowa
    - `salary`: Miesięczne wynagrodzenie (w tysiącach)
    - `credit_hist`: Historia kredytowa (1 = pozytywna, 0 = negatywna)
    - `other_credits`: (1 = klient posiada inne kredyty, 0 = klient nie posiada innych kredytów)
    - `target`: Zmienna docelowa (1 = przyznano kredyt, 0 = odmówiono kredytu)
    """)

    st.write("## Próbka danych")
    st.write(data.head())

    st.write("### Załadowanie danych:")
    st.code("""
    # Załadowanie danych
    @st.cache_data
    def load_data(file):
        return pd.read_excel(file)

    data = pd.DataFrame(load_data('./data_files/ms_data.xlsx')[['Age_group', 'mth_salary_tsd_group', 'positive_credit_history', 'other_credits', 'target']])
    data.rename(columns={'Age_group':'age', 'mth_salary_tsd_group':'salary', 'positive_credit_history':'credit_hist'}, inplace=True)
    """)

    # Funkcja do obliczania WOE
    def calculate_woe(df, variable, target, smoothing=0.5):
        grouped = df.groupby(variable)[target].agg(['sum', 'count'])
        grouped.columns = ['good', 'total']
        grouped['bad'] = grouped['total'] - grouped['good']
        
        # Dodanie smoothingu do liczby dobrych i złych klientów, aby uniknąć dzielenia przez zero
        grouped['good'] = grouped['good'] + smoothing
        grouped['bad'] = grouped['bad'] + smoothing

        # Obliczanie procentu dobrych i złych klientów
        grouped['%good'] = grouped['good'] / grouped['total'].sum()
        grouped['%bad'] = grouped['bad'] / grouped['total'].sum()

        # Obliczanie WOE jako ln(%good / %bad)
        grouped['WOE'] = np.log(grouped['%good'] / grouped['%bad'])
        
        return pd.DataFrame(grouped['WOE'])

    # Obliczanie Distribution Good i Distribution Bad
    def calc_distribution(data, feature, target):
        lst = []
        unique_values = data[feature].unique()
        for val in unique_values:
            df = data[data[feature] == val]
            good = len(df[df[target] == 0])
            bad = len(df[df[target] == 1])
            lst.append([feature, val, good, bad])
        
        df_dist = pd.DataFrame(lst, columns=['Variable', 'Value', 'Good', 'Bad'])
        
        total_good = df_dist['Good'].sum()
        total_bad = df_dist['Bad'].sum()
        df_dist['Distribution Good'] = df_dist['Good'] / total_good
        df_dist['Distribution Bad'] = df_dist['Bad'] / total_bad
        df_dist.rename(columns={'Value':feature}, inplace=True)
        
        return df_dist[[feature, 'Good', 'Bad', 'Distribution Good', 'Distribution Bad']]

    def calc_iv(df_dist):
        df_dist['IV'] = (df_dist['Distribution Good'] - df_dist['Distribution Bad']) * df_dist['WOE']
        iv = df_dist['IV'].sum()
        return iv

    # Obliczanie WOE
    st.write('## Waga dowodów (WOE):')

    st.write("""
    Waga dowodów (Weight of Evidence, WOE) jest miarą stosowaną w analizie danych i modelowaniu, szczególnie w kontekście oceny ryzyka kredytowego. WOE służy do przekształcania zmiennych kategorycznych na zmienne ciągłe, co czyni je bardziej odpowiednimi do modeli regresji logistycznej.

    WOE dla danej kategorii zmiennej jest obliczane jako logarytm naturalny stosunku odsetka pozytywnych zdarzeń (np. przyznanych kredytów) do odsetka negatywnych zdarzeń (np. odmówionych kredytów) w tej kategorii. Aby uniknąć problemów związanych z dzieleniem przez zero, do liczby pozytywnych i negatywnych zdarzeń dodawane jest tzw. smoothing. Wzór na WOE z uwzględnieniem smoothingu jest następujący:

    """
    + r"$\ln \left( \frac{\% \text{good + smoothing}}{\% \text{bad + smoothing}} \right)$" +
    """

    gdzie:
    - %good to procent pozytywnych zdarzeń w danej kategorii,
    - %bad to procent negatywnych zdarzeń w danej kategorii,
    - smoothing to mała stała wartość dodawana w celu uniknięcia dzielenia przez zero.

    WOE jest przydatne, ponieważ:
    1. Ułatwia wykrywanie zależności między zmiennymi niezależnymi a zmienną zależną.
    2. Umożliwia porównanie siły predykcyjnej różnych kategorii.
    3. Pomaga w identyfikacji i usunięciu zmiennych o niskiej wartości informacyjnej.

    Wartości WOE bliskie zeru sugerują, że dana kategoria nie ma istotnego wpływu na wynik. Wartości dodatnie wskazują na pozytywny wpływ, natomiast wartości ujemne na negatywny wpływ.
    """)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        age_woe_df = calculate_woe(data, 'age', 'target')
        st.write(age_woe_df)

    with col2:
        salary_woe_df = calculate_woe(data, 'salary', 'target')
        st.write(salary_woe_df)

    with col3:
        credit_hist_woe_df = calculate_woe(data, 'credit_hist', 'target')
        st.write(credit_hist_woe_df)

    with col4:
        other_credits_woe_df = calculate_woe(data, 'other_credits', 'target')
        st.write(other_credits_woe_df)

    st.write("### Funkcja do obliczania WOE:")
    st.code("""
    def calculate_woe(df, variable, target, smoothing=0.5):
        grouped = df.groupby(variable)[target].agg(['sum', 'count'])
        grouped.columns = ['good', 'total']
        grouped['bad'] = grouped['total'] - grouped['good']
        grouped['good'] = grouped['good'] + smoothing
        grouped['bad'] = grouped['bad'] + smoothing
        grouped['%good'] = grouped['good'] / grouped['total'].sum()
        grouped['%bad'] = grouped['bad'] / grouped['total'].sum()
        grouped['WOE'] = np.log(grouped['%good'] / grouped['%bad'])
        return pd.DataFrame(grouped['WOE'])
    """)

    # Oblicz Distribution Good i Distribution Bad dla grupowanych danych
    dist_age_df = calc_distribution(data, 'age', 'target')
    dist_salary_df = calc_distribution(data, 'salary', 'target')
    dist_credit_hist_df = calc_distribution(data, 'credit_hist', 'target')
    dist_other_credits_df = calc_distribution(data, 'other_credits', 'target')

    # Wyświetl wyniki
    st.write("## Rozkład dobrych i złych kredytów dla każdej grupy:")

    st.write('### Wiek:')
    st.write(dist_age_df)

    st.write("""
    Analizując rozkład dobrych i złych kredytów w różnych grupach wiekowych, możemy zauważyć kilka interesujących trendów:

    - Grupa wiekowa 18-25 ma najwyższy udział zarówno w dobrych (0.371), jak i złych kredytach (0.2987). Oznacza to, że młodsi klienci częściej otrzymują kredyty, ale także częściej mają problemy z ich spłatą.
    - Grupa wiekowa 25-35 również ma wysoki udział w dobrych kredytach (0.3594) i nieco niższy udział w złych kredytach (0.3054), co sugeruje, że są relatywnie bardziej wiarygodni w porównaniu do młodszych klientów.
    - Starsze grupy wiekowe (45-60 i 60+) mają niższy udział w dobrych kredytach (odpowiednio 0.0812 i 0.0203), ale stosunkowo wysoki udział w złych kredytach (odpowiednio 0.1577 i 0.0403), co może sugerować większe ryzyko kredytowe wśród starszych klientów.
    """)

    st.write('### Wynagrodzenie:')
    st.write(dist_salary_df)

    st.write("""
    Przyglądając się wynagrodzeniom, możemy wyciągnąć następujące wnioski:

    - Klienci zarabiający od 0 do 5 tysięcy złotych miesięcznie mają najwyższy udział w dobrych kredytach (0.7333), co może wskazywać na ich zdolność do regularnej spłaty zobowiązań.
    - W miarę wzrostu wynagrodzenia, udział w dobrych kredytach maleje. Na przykład, klienci zarabiający od 5 do 10 tysięcy złotych mają udział 0.1217 w dobrych kredytach.
    - Najwyższy udział w złych kredytach mają klienci zarabiający od 5 do 10 tysięcy złotych (0.2416), co może sugerować, że ta grupa napotyka na trudności w zarządzaniu większymi zobowiązaniami.
    - Klienci zarabiający powyżej 20 tysięcy złotych mają najniższy udział w złych kredytach, co może świadczyć o ich lepszej zdolności kredytowej.
    """)

    st.write('### Historia kredytowa:')
    st.write(dist_credit_hist_df)

    st.write("""
    Historia kredytowa ma istotny wpływ na jakość kredytów:

    - Klienci z pozytywną historią kredytową mają znacznie wyższy udział w dobrych kredytach (0.6812) w porównaniu do tych z negatywną historią (0.3188).
    - Udział w złych kredytach jest znacznie wyższy dla klientów z pozytywną historią kredytową (0.8523) w porównaniu do tych z negatywną historią (0.1477). Może to być zaskakujące, ale sugeruje, że nawet klienci z pozytywną historią mogą mieć trudności z nowymi zobowiązaniami.
    """)

    st.write('### Inne kredyty:')
    st.write(dist_other_credits_df)

    st.write("""
    Posiadanie innych kredytów również wpływa na ryzyko kredytowe:

    - Klienci bez innych kredytów mają znacznie wyższy udział w dobrych kredytach (0.7855) w porównaniu do tych posiadających inne kredyty (0.2145).
    - Udział w złych kredytach jest znacznie wyższy dla klientów posiadających inne kredyty (0.6846), co sugeruje, że wielokrotne zobowiązania mogą zwiększać ryzyko niewypłacalności.
    - Klienci bez innych kredytów mają znacznie niższy udział w złych kredytach (0.3154), co wskazuje na ich większą zdolność do terminowej spłaty zobowiązań.
    """)

    st.write("### Obliczanie Distribution Good i Distribution Bad:")
    st.code("""
    def calc_distribution(data, feature, target):
        lst = []
        unique_values = data[feature].unique()
        for val in unique_values:
            df = data[data[feature] == val]
            good = len(df[df[target] == 0])
            bad = len(df[df[target] == 1])
            lst.append([feature, val, good, bad])
        
        df_dist = pd.DataFrame(lst, columns=['Variable', 'Value', 'Good', 'Bad'])
        
        total_good = df_dist['Good'].sum()
        total_bad = df_dist['Bad'].sum()
        df_dist['Distribution Good'] = df_dist['Good'] / total_good
        df_dist['Distribution Bad'] = df_dist['Bad'] / total_bad
        df_dist.rename(columns={'Value':feature}, inplace=True)
        
        return df_dist[[feature, 'Good', 'Bad', 'Distribution Good', 'Distribution Bad']]
            
    dist_age_df = calc_distribution(data, 'age', 'target')
    dist_salary_df = calc_distribution(data, 'salary', 'target')
    dist_credit_hist_df = calc_distribution(data, 'credit_hist', 'target')
    dist_other_credits_df = calc_distribution(data, 'other_credits', 'target')
    """)

    # Złączamyt WOE i distribution
    merged_age = pd.merge(age_woe_df, dist_age_df, on='age')
    merged_salary = pd.merge(salary_woe_df, dist_salary_df, on='salary')
    merged_credit_history = pd.merge(credit_hist_woe_df, dist_credit_hist_df, on='credit_hist')
    merged_other_credits = pd.merge(other_credits_woe_df, dist_other_credits_df, on='other_credits')

    # Obliczamy IV
    iv_age = calc_iv(merged_age)
    iv_salary = calc_iv(merged_salary)
    iv_credit_history = calc_iv(merged_credit_history)
    iv_other_credits = calc_iv(merged_other_credits)

    st.write(f'## Wartość informacyjna (IV):')
    st.write(f'IV_age: {round(iv_age,4)}')
    st.write(f'IV_salary: {round(iv_salary,4)}')
    st.write(f'IV_credit_history: {round(iv_credit_history,4)}')
    st.write(f'IV_other_credits: {round(iv_other_credits,4)}')

    st.write("""
    Wartość informacyjna (Information Value, IV) jest miarą stosowaną do oceny siły predykcyjnej zmiennej w kontekście modelowania ryzyka kredytowego. IV pomaga w identyfikacji, które zmienne mają największy wpływ na wynik modelu. Im wyższa wartość IV, tym większa jest moc predykcyjna zmiennej. Wartości IV są interpretowane według następujących kryteriów:

    - IV < 0.02: Zmienna nieistotna
    - 0.02 <= IV < 0.1: Zmienna o niskiej predykcyjności
    - 0.1 <= IV < 0.3: Zmienna o średniej predykcyjności
    - IV >= 0.3: Zmienna o wysokiej predykcyjności

    ### Interpretacja:

    - **IV_age (-0.0926)**:
    Wartość informacyjna dla wieku jest ujemna i bliska zeru, co wskazuje, że wiek ma bardzo niską moc predykcyjną. Może to sugerować, że wiek nie jest istotnym czynnikiem przy ocenie ryzyka kredytowego w tym modelu.

    - **IV_salary (-0.2233)**:
    Wartość informacyjna dla wynagrodzenia jest ujemna i wynosi -0.2233, co sugeruje, że wynagrodzenie ma średnią moc predykcyjną. Wynagrodzenie jest istotnym czynnikiem, ale jego wpływ na wynik modelu nie jest bardzo silny.

    - **IV_credit_history (-0.169)**:
    Wartość informacyjna dla historii kredytowej wynosi -0.169, co oznacza, że historia kredytowa ma średnią moc predykcyjną. Jest to istotny czynnik przy ocenie ryzyka kredytowego, ale nie dominuje w modelu.

    - **IV_other_credits (-0.9708)**:
    Wartość informacyjna dla liczby innych kredytów jest ujemna i wynosi -0.9708, co sugeruje, że liczba innych kredytów ma bardzo wysoką moc predykcyjną. To oznacza, że posiadanie innych kredytów jest kluczowym czynnikiem wpływającym na ryzyko kredytowe i powinno być brane pod uwagę przy podejmowaniu decyzji kredytowych.

    Podsumowując, analiza wartości informacyjnych (IV) dla poszczególnych zmiennych pozwala zidentyfikować, które czynniki mają największy wpływ na przewidywanie ryzyka kredytowego. W tym przypadku, liczba innych kredytów okazała się być najważniejszą zmienną, podczas gdy wiek ma najmniejszy wpływ na wynik modelu.
    """)

    st.write("### Obliczanie IV:")
    st.code("""
    def calc_iv(df_dist):
        df_dist['IV'] = (df_dist['Distribution Good'] - df_dist['Distribution Bad']) * df_dist['WOE']
        iv = df_dist['IV'].sum()
        return iv

    iv_age = calc_iv(merged_age)
    iv_salary = calc_iv(merged_salary)
    iv_credit_history = calc_iv(merged_credit_history)
    iv_other_credits = calc_iv(merged_other_credits)
    """)

    # Zmieniamy nazwy kolumn WOE, żeby uniknąć tych samych nazw po złączeniu
    merged_age.rename(columns={'WOE':'WOE_age'}, inplace=True)
    merged_salary.rename(columns={'WOE':'WOE_salary'}, inplace=True)
    merged_credit_history.rename(columns={'WOE':'WOE_credit_hist'}, inplace=True)
    merged_other_credits.rename(columns={'WOE':'WOE_other_credits'}, inplace=True)

    #Złączamy dane początkowe i WOE

    data = pd.merge(data, merged_age[['age','WOE_age']], on='age')
    data = pd.merge(data, merged_salary[['salary','WOE_salary']], on='salary')
    data = pd.merge(data, merged_credit_history[['credit_hist','WOE_credit_hist']], on='credit_hist')
    data = pd.merge(data, merged_other_credits[['WOE_other_credits','other_credits']], on='other_credits')

    st.write("## Dane z przypisanymi wartościami WOE:")
    st.write(data)

    # Tworzymy histogramy
    st.write("## Histogramy:")

    st.write("""
     **Histogramy rozkładu**: Te histogramy pokazują rozkład każdej cechy w zestawie danych, dając wyobrażenie o rozkładzie danych i możliwych skośnościach.
    """)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs[0, 0].hist(data['age'], bins=20, alpha=0.7, label='Wiek')
    axs[0, 0].set_title('Rozkład wieku')

    axs[0, 1].hist(data['salary'], bins=20, alpha=0.7, label='Wynagrodzenie')
    axs[0, 1].set_title('Rozkład wynagrodzenia')

    axs[1, 0].hist(data['credit_hist'], bins=20, alpha=0.7, label='Historia kredytowa')
    axs[1, 0].set_title('Rozkład historii kredytowej')
    axs[1, 0].set_xticks([0, 1])

    axs[1, 1].hist(data['other_credits'], bins=20, alpha=0.7, label='Inne kredyty')
    axs[1, 1].set_title('Rozkład innych kredytów')
    axs[1, 1].set_xticks([0, 1])

    for ax in axs.flat:
        ax.set_xlabel('Wartość')
        ax.set_ylabel('Częstotliwość')
        ax.legend()

    st.pyplot(fig)

    # Przygotowujemy dane pod model, podział na macierz X i y
    X = data[['WOE_age', 'WOE_salary', 'WOE_credit_hist', 'WOE_other_credits']]
    y = data['target']

    st.write('## Przygotowanie danych pod model:')
    st.code("""
    X = data[['WOE_age', 'WOE_salary', 'WOE_credit_hist', 'WOE_other_credits']]
    y = data['target']
    """)

    # Podział na zbiory: treningowy, walidacyjny i testowy
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2


    st.write("## Podział danych na zbiory testowe, trenujące i walidacyjne:")
    st.write(f"Rozmiar zestawu treningowego: {X_train.shape} - {100*round(X_train.shape[0]/X.shape[0],2)}% zbioru X.")
    st.write(f"Rozmiar zestawu walidacyjnego: {X_val.shape} - {100*round(X_val.shape[0]/X.shape[0],2)}% zbioru X.")
    st.write(f"Rozmiar zestawu testowego: {X_test.shape} - {100*round(X_test.shape[0]/X.shape[0],2)}% zbioru X.")

    st.code("""
    X = data[['WOE_age', 'WOE_salary', 'WOE_credit_hist', 'WOE_other_credits']]
    y = data['target']

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2
    """)

    st.write("""
    ### Interpretacja:

    - **Zestaw treningowy (60%)**:
    Zestaw treningowy składa się z 385 próbek, co stanowi 60% całego zbioru danych. Jest to zestaw używany do trenowania modelu, czyli do uczenia modelu na podstawie dostarczonych danych. Większość danych jest używana w tym celu, aby model mógł nauczyć się jak najwięcej informacji z dostępnych danych.

    - **Zestaw walidacyjny (20%)**:
    Zestaw walidacyjny składa się z 129 próbek, co stanowi 20% całego zbioru danych. Jest to zestaw używany do oceny modelu podczas procesu trenowania. Na podstawie wyników uzyskanych na tym zbiorze, można dostroić hiperparametry modelu oraz ocenić jego wydajność, zanim zostanie przetestowany na zestawie testowym. Walidacja pozwala na uniknięcie przeuczenia modelu (overfitting).

    - **Zestaw testowy (20%)**:
    Zestaw testowy składa się z 129 próbek, co stanowi 20% całego zbioru danych. Jest to zestaw używany do ostatecznej oceny modelu po zakończeniu procesu trenowania i walidacji. Wyniki uzyskane na tym zbiorze odzwierciedlają rzeczywistą wydajność modelu na nieznanych wcześniej danych. Jest to kluczowy krok w procesie ewaluacji, ponieważ pozwala na ocenę, jak dobrze model generalizuje na nowe dane.

    Podział danych na te trzy zbiory jest istotny, ponieważ pozwala na rzetelną ocenę wydajności modelu i zapewnia, że model nie jest dopasowany wyłącznie do danych treningowych, ale potrafi także dobrze przewidywać na nowych, nieznanych danych.
    """)

    # Trenujemy model regresji logistycznej
    model = LogisticRegression()
    model.fit(X_train, y_train)

    st.write("## Trening modelu regresji logistycznej:")
    st.code("""
    model = LogisticRegression()
    model.fit(X_train, y_train)
    """)

    # Wyciągnięcie współczynników
    coefficients = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_[0]
    })

    # Dodanie interceptu do tabeli
    intercept = pd.DataFrame({
        'Feature': ['Intercept'],
        'Coefficient': [model.intercept_[0]]
    })

    coefficients = pd.concat([intercept, coefficients], ignore_index=True)

    # Wyświetlenie tabeli współczynników
    st.write("## Współczynniki modelu regresji logistycznej")
    st.write(coefficients)

    st.write("""
    ### Interpretacja:

    - **Intercept (0.27)**:
    Intercept, czyli wyraz wolny w modelu regresji logistycznej, ma wartość 0.27. Oznacza to, że gdy wszystkie zmienne objaśniające (WOE_age, WOE_salary, WOE_credit_hist, WOE_other_credits) są równe zero, logarytm szansy na przyznanie kredytu wynosi 0.27. W praktyce, wartość interceptu przesuwa całkowity poziom log-odds w modelu.

    - **WOE_age (0.3133)**:
    Współczynnik dla WOE_age wynosi 0.3133. Wartość ta sugeruje, że wiek (po przekształceniu na WOE) ma pozytywny wpływ na log-odds przyznania kredytu. Każdy wzrost WOE_age o jednostkę powoduje wzrost log-odds przyznania kredytu o 0.3133, przy założeniu, że pozostałe zmienne pozostają bez zmian. Wiek jest więc istotnym, choć nie najsilniejszym czynnikiem w modelu.

    - **WOE_salary (0.7683)**:
    Współczynnik dla WOE_salary wynosi 0.7683. Oznacza to, że miesięczne wynagrodzenie (po przekształceniu na WOE) ma znaczący pozytywny wpływ na log-odds przyznania kredytu. Każdy wzrost WOE_salary o jednostkę powoduje wzrost log-odds przyznania kredytu o 0.7683. Wyższe wynagrodzenie zwiększa więc szanse na otrzymanie kredytu.

    - **WOE_credit_hist (0.4603)**:
    Współczynnik dla WOE_credit_hist wynosi 0.4603. Wskazuje to, że pozytywna historia kredytowa (po przekształceniu na WOE) ma pozytywny wpływ na log-odds przyznania kredytu. Każdy wzrost WOE_credit_hist o jednostkę powoduje wzrost log-odds przyznania kredytu o 0.4603. Dobra historia kredytowa jest więc istotnym czynnikiem w modelu.

    - **WOE_other_credits (0.9185)**:
    Współczynnik dla WOE_other_credits wynosi 0.9185. Jest to najwyższa wartość spośród wszystkich zmiennych, co sugeruje, że liczba innych kredytów (po przekształceniu na WOE) ma najsilniejszy pozytywny wpływ na log-odds przyznania kredytu. Każdy wzrost WOE_other_credits o jednostkę powoduje wzrost log-odds przyznania kredytu o 0.9185. Posiadanie innych kredytów jest więc kluczowym czynnikiem zwiększającym szanse na otrzymanie nowego kredytu.

    ### Co to są log-odds?

    Log-odds (logarytm szans) to pojęcie używane w regresji logistycznej do wyrażania stosunku prawdopodobieństw. W kontekście modelu regresji logistycznej, log-odds są logarytmem naturalnym stosunku prawdopodobieństwa wystąpienia zdarzenia do prawdopodobieństwa jego niewystąpienia.

    Matematycznie, log-odds są wyrażone jako:
    """ + r"$\text{log-odds} = \ln\left(\frac{P}{1-P}\right)$" + """
    gdzie \( P \) jest prawdopodobieństwem wystąpienia zdarzenia (np. przyznania kredytu).

    Wartości log-odds mogą być przekształcone z powrotem na prawdopodobieństwa za pomocą funkcji logistycznej:
    """ + r"$P = \frac{e^{\text{log-odds}}}{1 + e^{\text{log-odds}}}$" + """
    
    Podsumowując, wszystkie współczynniki w modelu są dodatnie, co oznacza, że wzrost wartości każdej z tych zmiennych (po przekształceniu na WOE) zwiększa log-odds przyznania kredytu. W szczególności, liczba innych kredytów ma najsilniejszy wpływ, co sugeruje, że instytucje finansowe szczególnie biorą pod uwagę zdolność klienta do zarządzania wieloma zobowiązaniami przy ocenie ryzyka kredytowego.
            
    """)

    # Predykcja na zbiorze walidacyjnym
    y_val_pred_proba = model.predict_proba(X_val)[:, 1]
    y_val_pred = model.predict(X_val)

    # Ewaluacja modelu na zbiorze walidacyjnym
    roc_auc_val = roc_auc_score(y_val, y_val_pred_proba)
    accuracy_val = accuracy_score(y_val, y_val_pred)

    st.write("## Ewaluacja modelu na zbiorze walidacyjnym")
    st.write(f"ROC AUC: {round(roc_auc_val,4)}")
    st.write(f"Accuracy: {round(accuracy_val,4)}")

    st.write("""
    ### Interpretacja:

    - **ROC AUC (0.778)**:
    ROC AUC (Receiver Operating Characteristic Area Under the Curve) jest miarą zdolności modelu do rozróżniania między klasami pozytywnymi i negatywnymi. Wartość 0.778 wskazuje, że model ma dobrą zdolność do klasyfikacji. Im bliżej wartości 1, tym lepsza wydajność modelu. Wynik 0.778 sugeruje, że model skutecznie odróżnia klientów, którym należy przyznać kredyt, od tych, którym kredyt nie powinien być przyznany.

    - **Accuracy (0.7442)**:
    Accuracy (dokładność) to miara określająca procent poprawnych przewidywań dokonanych przez model. Wartość 0.7442 oznacza, że model prawidłowo przewiduje przyznanie lub odmowę kredytu w około 74.42% przypadków. Jest to dobry wynik, choć należy pamiętać, że accuracy może być mniej informatywne w przypadku niezrównoważonych zbiorów danych.


    Wyniki ewaluacji modelu na zbiorze walidacyjnym wskazują, że model regresji logistycznej osiąga dobrą wydajność w przewidywaniu decyzji kredytowych. Wartość ROC AUC wynosząca 0.778 pokazuje, że model jest skuteczny w rozróżnianiu między klasami, a dokładność na poziomie 0.7442 sugeruje, że model dobrze radzi sobie z przewidywaniem decyzji kredytowych. Te wyniki sugerują, że model może być wartościowym narzędziem wspomagającym proces decyzyjny w przyznawaniu kredytów.
    """)

    st.code("""
    y_val_pred_proba = model.predict_proba(X_val)[:, 1]
    y_val_pred = model.predict(X_val)

    roc_auc_val = roc_auc_score(y_val, y_val_pred_proba)
    accuracy_val = accuracy_score(y_val, y_val_pred)
    """)

    # Predykcja na zbiorze testowym
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # Ewaluacja modelu na zbiorze testowym
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)

    st.write("## Ewaluacja modelu na zbiorze testowym")
    st.write(f"ROC AUC: {round(roc_auc,4)}")
    st.write(f"Accuracy: {round(accuracy,4)}")

    st.write("""
    ### Interpretacja:

    - **ROC AUC (0.8212)**:
    ROC AUC (Receiver Operating Characteristic Area Under the Curve) jest miarą zdolności modelu do rozróżniania między klasami pozytywnymi i negatywnymi. Wartość 0.8212 wskazuje, że model ma bardzo dobrą zdolność do klasyfikacji. Im bliżej wartości 1, tym lepsza wydajność modelu. Wynik 0.8212 sugeruje, że model skutecznie odróżnia klientów, którym należy przyznać kredyt, od tych, którym kredyt nie powinien być przyznany. Lepszy wynik ROC AUC na zbiorze testowym w porównaniu do zbioru walidacyjnego (0.778) wskazuje na dobrą generalizację modelu na nowych danych.

    - **Accuracy (0.7519)**:
    Accuracy (dokładność) to miara określająca procent poprawnych przewidywań dokonanych przez model. Wartość 0.7519 oznacza, że model prawidłowo przewiduje przyznanie lub odmowę kredytu w około 75.19% przypadków. Jest to nieco wyższy wynik niż na zbiorze walidacyjnym (0.7442), co sugeruje, że model nie jest przeuczony i dobrze radzi sobie z nowymi danymi.


    Wyniki ewaluacji modelu na zbiorze testowym wskazują, że model regresji logistycznej osiąga bardzo dobrą wydajność w przewidywaniu decyzji kredytowych. Wartość ROC AUC wynosząca 0.8212 pokazuje, że model jest skuteczny w rozróżnianiu między klasami, a dokładność na poziomie 0.7519 sugeruje, że model dobrze radzi sobie z przewidywaniem decyzji kredytowych. Te wyniki potwierdzają, że model może być wartościowym narzędziem wspomagającym proces decyzyjny w przyznawaniu kredytów, oferując solidne i niezawodne przewidywania.
    """)

    st.code("""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    """)

    # Wyliczenie macierzy konfuzji
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Wizualizacja macierzy konfuzji
    st.write("## Macierz konfuzji")
    fig, ax = plt.subplots()
    cax = ax.matshow(conf_matrix, cmap='Blues')
    plt.colorbar(cax)

    # Dodanie wartości do komórek macierzy konfuzji
    for (i, j), val in np.ndenumerate(conf_matrix):
        ax.text(j, i, f'{val}', ha='center', va='center', color='black')

    ax.set_xlabel('Przewidywane wartości')
    ax.set_ylabel('Rzeczywiste wartości')
    ax.set_title('Macierz konfuzji')
    ax.set_xticks(range(len(conf_matrix)))
    ax.set_yticks(range(len(conf_matrix)))

    st.pyplot(fig)

    st.write("""
    ### Interpretacja:

    Macierz konfuzji pokazuje, jak dobrze model radzi sobie z klasyfikacją danych na zbiorze testowym. Jest to narzędzie do oceny wydajności modelu klasyfikacyjnego, które prezentuje liczbę prawdziwych i fałszywych pozytywów oraz negatywów. Wartości w macierzy konfuzji dla tego modelu są następujące:

    - **True Negatives (TN, Prawdziwe negatywy)**: 52
    - Liczba przypadków, w których model poprawnie przewidział, że kredyt nie zostanie przyznany (0).
    - **False Positives (FP, Fałszywe pozytywy)**: 20
    - Liczba przypadków, w których model błędnie przewidział, że kredyt zostanie przyznany (1), podczas gdy w rzeczywistości kredyt nie został przyznany (0).
    - **False Negatives (FN, Fałszywe negatywy)**: 12
    - Liczba przypadków, w których model błędnie przewidział, że kredyt nie zostanie przyznany (0), podczas gdy w rzeczywistości kredyt został przyznany (1).
    - **True Positives (TP, Prawdziwe pozytywy)**: 45
    - Liczba przypadków, w których model poprawnie przewidział, że kredyt zostanie przyznany (1).

    ### Metryki z macierzy konfuzji:

    Na podstawie macierzy konfuzji można obliczyć kilka ważnych metryk:

    - **Accuracy (Dokładność)**:
    - Accuracy = (TP + TN) / (TP + TN + FP + FN)
    - Dokładność wynosi (52 + 45) / (52 + 45 + 20 + 12) = 0.7519, co oznacza, że model poprawnie klasyfikuje około 75.19% przypadków.

    - **Precision (Precyzja)**:
    - Precision = TP / (TP + FP)
    - Precyzja wynosi 45 / (45 + 20) = 0.6923, co oznacza, że spośród wszystkich przypadków przewidzianych jako pozytywne, około 69.23% jest rzeczywiście pozytywnych.

    - **Recall (Czułość)**:
    - Recall = TP / (TP + FN)
    - Czułość wynosi 45 / (45 + 12) = 0.7895, co oznacza, że model wykrywa około 78.95% wszystkich rzeczywistych pozytywnych przypadków.

    - **F1 Score**:
    - F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    - F1 Score wynosi 2 * (0.6923 * 0.7895) / (0.6923 + 0.7895) = 0.7386, co stanowi harmoniczną średnią precyzji i czułości.

    Macierz konfuzji pokazuje, że model ma dobrą zdolność do klasyfikacji, ale nadal istnieją pewne błędy klasyfikacji. Wartości True Positives i True Negatives są wysokie, co wskazuje na wysoką dokładność modelu. Model ma jednak pewną liczbę False Positives i False Negatives, co sugeruje, że istnieje możliwość dalszej optymalizacji modelu. Ogólnie rzecz biorąc, wyniki są zachęcające i sugerują, że model jest skuteczny w przewidywaniu przyznawania kredytów.
    """)

    st.code("""
    conf_matrix = confusion_matrix(y_test, y_pred)
    """)

    # Rysowanie krzywej ROC dla zbioru testowego
    st.write('\n')
    st.write("## Krzywa ROC")
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'Krzywa ROC (powierzchnia = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Wskaźnik fałszywie pozytywny')
    plt.ylabel('Wskaźnik prawdziwie pozytywny')
    plt.title('Odbiorcza Krzywa Charakterystyczna')
    plt.legend(loc="lower right")

    st.pyplot(plt)

    st.write("""
    ### Krzywa ROC:

    Krzywa ROC (Receiver Operating Characteristic) przedstawia zdolność modelu do rozróżniania między klasami pozytywnymi i negatywnymi. Na osi poziomej (X) znajduje się wskaźnik fałszywie pozytywny (False Positive Rate), a na osi pionowej (Y) wskaźnik prawdziwie pozytywny (True Positive Rate). Krzywa ROC pozwala na ocenę wydajności modelu klasyfikacyjnego przy różnych progach decyzyjnych.

    ### Interpretacja:

    - **Krzywa ROC**:
    Krzywa ROC modelu pokazuje, jak zmienia się wskaźnik prawdziwie pozytywny (TPR) w funkcji wskaźnika fałszywie pozytywnego (FPR) przy różnych progach decyzyjnych. Model osiąga bardzo dobrą wydajność, co jest widoczne po tym, że krzywa jest daleko od linii losowej (przerywana linia diagonalna).

    - **Powierzchnia pod krzywą (AUC)**:
    Wartość AUC (Area Under the Curve) wynosi 0.82, co oznacza, że model ma wysoką zdolność do rozróżniania między klasami. Wartość AUC bliska 1 wskazuje na doskonały model, podczas gdy wartość 0.5 sugeruje model losowy. Wartość 0.82 wskazuje na bardzo dobrą wydajność modelu w przewidywaniu przyznawania kredytów.

    - **Wskaźnik prawdziwie pozytywny (TPR)**:
    TPR (True Positive Rate) to stosunek liczby prawidłowo przewidzianych pozytywnych przypadków do wszystkich rzeczywistych pozytywnych przypadków. Jest również nazywany czułością (sensitivity). Wyższy TPR wskazuje, że model dobrze identyfikuje pozytywne przypadki.

    - **Wskaźnik fałszywie pozytywny (FPR)**:
    FPR (False Positive Rate) to stosunek liczby błędnie przewidzianych pozytywnych przypadków do wszystkich rzeczywistych negatywnych przypadków. Niższy FPR wskazuje, że model rzadziej popełnia błędy klasyfikując negatywne przypadki jako pozytywne.

    Krzywa ROC i wartość AUC dostarczają ważnych informacji na temat wydajności modelu klasyfikacyjnego. W tym przypadku, wartość AUC wynosząca 0.82 sugeruje, że model regresji logistycznej ma wysoką zdolność do rozróżniania między klientami, którzy powinni otrzymać kredyt, a tymi, którzy nie powinni. Model jest skuteczny i może być z powodzeniem stosowany w procesie decyzyjnym przyznawania kredytów.
    """)

    st.code("""
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    """)

    # Cross walidacja
    st.write("## Cross walidacja")
    cv_scores = cross_val_score(model, X_train, y_train, cv=10, scoring='roc_auc')
    st.write(f'Średni wynik ROC AUC z cross walidacji: {np.mean(cv_scores):.4f}')

    st.write("""
    ### Interpretacja:

    Cross walidacja to technika oceny wydajności modelu polegająca na podziale danych na wiele podzbiorów (folds) i trenowaniu oraz testowaniu modelu na tych podzbiorach. W tym przypadku użyto cross walidacji z 10 podzbiorami (10-fold cross-validation), co oznacza, że dane zostały podzielone na 10 równych części. Model był trenowany na 9 częściach i testowany na pozostałej części, a proces ten był powtarzany 10 razy, za każdym razem używając innego podzbioru jako zbioru testowego.

    Średni wynik ROC AUC z cross walidacji wynosi 0.7765, co sugeruje, że model ma stabilną i dobrą wydajność na różnych podzbiorach danych.

    - **ROC AUC (Receiver Operating Characteristic Area Under the Curve)**:
    Wartość ROC AUC mierzy zdolność modelu do rozróżniania między klasami pozytywnymi i negatywnymi. Średni wynik ROC AUC wynoszący 0.7765 oznacza, że model jest w stanie dobrze klasyfikować przypadki na różnych podzbiorach danych. Wynik ten jest zbliżony do wyników uzyskanych na zbiorze walidacyjnym i testowym, co wskazuje na spójność i niezawodność modelu.

    Średni wynik ROC AUC z cross walidacji wynoszący 0.7765 pokazuje, że model regresji logistycznej ma stabilną i dobrą zdolność do rozróżniania między klasami na różnych podzbiorach danych. Cross walidacja dostarcza bardziej wiarygodnej oceny wydajności modelu, ponieważ uwzględnia zmienność wyników na różnych częściach danych. Wysoki i spójny wynik ROC AUC sugeruje, że model jest dobrze dostosowany i może być z powodzeniem stosowany w praktyce do przewidywania przyznawania kredytów.
    """)

    st.code("""
    cv_scores = cross_val_score(model, X_train, y_train, cv=10, scoring='roc_auc')
    st.write(f'Średni wynik ROC AUC z cross walidacji: {np.mean(cv_scores):.4f}')
    """)

    st.write("""
    ### Podsumowanie

    W ramach tego projektu przeprowadzono szczegółową analizę i ewaluację modelu regresji logistycznej do przewidywania przyznawania kredytów. Wykorzystano dane zawierające informacje o klientach, takie jak grupa wiekowa, miesięczne wynagrodzenie, historia kredytowa oraz liczba posiadanych innych kredytów. Model został przekształcony przy użyciu wag dowodów (WOE) i oceniony pod kątem różnych metryk wydajności.

    ### Kluczowe wnioski:

    1. **Ewaluacja modelu**:
    - Model osiągnął wysoki wynik ROC AUC zarówno na zbiorze walidacyjnym (0.778), jak i testowym (0.8212), co wskazuje na jego zdolność do skutecznego rozróżniania między klientami, którym należy przyznać kredyt, a tymi, którym kredyt nie powinien być przyznany.
    - Dokładność modelu wynosiła odpowiednio 74.42% na zbiorze walidacyjnym i 75.19% na zbiorze testowym, co potwierdza jego solidną wydajność.

    2. **Macierz konfuzji**:
    - Analiza macierzy konfuzji ujawniła, że model ma wysoką liczbę prawdziwie pozytywnych i prawdziwie negatywnych klasyfikacji, ale także pewne błędy klasyfikacyjne w postaci fałszywie pozytywnych i fałszywie negatywnych wyników.

    3. **Krzywa ROC**:
    - Krzywa ROC i wartość AUC (0.82) potwierdzają wysoką wydajność modelu w zakresie rozróżniania między klasami. Krzywa ROC jest daleko od linii losowej, co sugeruje dobrą zdolność klasyfikacyjną modelu.

    4. **Cross walidacja**:
    - Średni wynik ROC AUC z cross walidacji wynoszący 0.7765 pokazuje, że model ma stabilną i dobrą wydajność na różnych podzbiorach danych, co potwierdza jego niezawodność.

    5. **Wartości informacyjne (IV)**:
    - Analiza wartości informacyjnych ujawniła, że zmienne takie jak historia kredytowa i liczba innych kredytów mają największą moc predykcyjną. Wiek okazał się najmniej istotnym czynnikiem.

    ### Wnioski praktyczne:

    - **Zastosowanie modelu**:
    Model regresji logistycznej może być skutecznie wykorzystany w instytucjach finansowych do wspomagania procesu decyzyjnego przyznawania kredytów. Dzięki wysokiej zdolności predykcyjnej modelu, instytucje mogą lepiej zarządzać ryzykiem kredytowym i podejmować bardziej świadome decyzje.

    - **Dalsza optymalizacja**:
    Chociaż model osiągnął dobre wyniki, istnieje możliwość dalszej optymalizacji, na przykład poprzez uwzględnienie dodatkowych zmiennych lub zastosowanie bardziej zaawansowanych technik modelowania.

    Podsumowując, model regresji logistycznej okazał się wartościowym narzędziem w przewidywaniu ryzyka kredytowego, oferując solidne i niezawodne wyniki. Jego zastosowanie może znacząco wspomóc procesy decyzyjne w sektorze finansowym.
    """)

with tkd:
    # Wstęp
    st.header("Wstęp")
    st.write("""
    W tym projekcie stworzymy model konwolucyjnej sieci neuronowej (CNN) do klasyfikacji obrazów ręcznie pisanych cyfr z zestawu danych MNIST.
    CNN to popularna metoda deep learningowa, która jest szczególnie skuteczna w przetwarzaniu obrazów.
    """)

    # Jak działa CNN
    st.header("Jak działa Konwolucyjna Sieć Neuronowa (CNN)?")
    st.write("""
    Konwolucyjna Sieć Neuronowa (CNN) jest typem sztucznej sieci neuronowej, która jest szczególnie skuteczna w przetwarzaniu obrazów. Oto prosty opis, jak działa CNN:
    """)
    st.write("""
    1. **Warstwa konwolucyjna**:
    - Główna idea warstwy konwolucyjnej polega na przesuwaniu filtrów (małych macierzy) po obrazie wejściowym.
    - Każdy filtr wykrywa różne cechy obrazu, takie jak krawędzie, tekstury czy kolory.
    - Wynikiem tej operacji jest mapa cech, która pokazuje, gdzie na obrazie występują wykryte cechy.
    """)
    st.write("""
    2. **Warstwa pooling**:
    - Po warstwie konwolucyjnej następuje warstwa pooling (najczęściej max pooling).
    - Warstwa ta zmniejsza rozmiar mapy cech, uśredniając lub wybierając maksymalne wartości z małych regionów mapy.
    - Pooling pomaga w redukcji wymiarów danych i zwiększa odporność na przesunięcia i zniekształcenia obrazu.
    """)
    st.write("""
    3. **Warstwy w pełni połączone (dense)**:
    - Po kilku warstwach konwolucyjnych i pooling, mapa cech jest przekształcana w jednowymiarowy wektor i podawana do warstw w pełni połączonych.
    - Warstwy te działają jak klasyczne sieci neuronowe, przetwarzając wektor cech i ucząc się klasyfikować obraz na podstawie cech wykrytych w poprzednich warstwach.
    """)
    st.write("""
    4. **Funkcja aktywacji**:
    - Funkcje aktywacji, takie jak ReLU (Rectified Linear Unit), są stosowane po każdej warstwie konwolucyjnej i dense, aby wprowadzić nieliniowość i umożliwić sieci uczenie się bardziej złożonych wzorców.
    """)
    st.write("""
    5. **Funkcja straty i optymalizacja**:
    - CNN jest trenowana przy użyciu funkcji straty, która mierzy różnicę między przewidywaniami modelu a rzeczywistymi etykietami.
    - Optymalizator, taki jak Adam, aktualizuje wagi sieci, aby minimalizować funkcję straty.
    """)
    st.write("""
    ### Podsumowanie
    Konwolucyjna Sieć Neuronowa (CNN) składa się z warstw konwolucyjnych, pooling i dense, które razem uczą się wykrywać i klasyfikować cechy obrazu. CNN są potężnym narzędziem do analizy obrazów, które znajdują szerokie zastosowanie w rozpoznawaniu obiektów, detekcji i innych zadaniach związanych z wizją komputerową.
    """)

    # Ładowanie danych
    st.header("Ładowanie danych")
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    st.code("""
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    """, language='python')

    # Wyświetlenie przykładowych obrazów
    st.subheader("Przykładowe obrazy")
    fig, axes = plt.subplots(1, 20, figsize=(20, 5))
    for i in range(20):
        axes[i].imshow(train_images[i], cmap='gray')
        axes[i].axis('off')
    st.pyplot(fig)

    # Przeskalowanie obrazów do zakresu 0-1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Dodanie dodatkowego wymiaru (głębi kanału)
    train_images = train_images[..., tf.newaxis]
    test_images = test_images[..., tf.newaxis]

    # Definiowanie modelu
    st.header("Definiowanie modelu")
    st.write("""
    Model konwolucyjnej sieci neuronowej składa się z kilku warstw konwolucyjnych, warstw zmniejszających wymiary (pooling), a następnie warstw w pełni połączonych (dense).
    Model jest kompilowany z użyciem optymalizatora Adam i funkcji straty sparse_categorical_crossentropy.
    """)
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    st.code("""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    """, language='python')

    # Kompilacja modelu
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # Trenowanie modelu z paskiem postępu
    st.header("Trenowanie modelu")

    st.code("""
    epochs = 5

    history = model.fit(train_images, train_labels, epochs=epochs, 
                        validation_data=(test_images, test_labels), 
                        callbacks=[StreamlitCallback()])
    """, language='python')

    epochs = 5
    progress_bar = st.progress(0)
    status_text = st.empty()

    class StreamlitCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            progress_bar.progress((epoch + 1) / epochs)
            status_text.text(f"Epoka: {epoch + 1}/{epochs}, Dokładność: {logs['accuracy']:.4f}, Walidacja dokładność: {logs['val_accuracy']:.4f}")

    history = model.fit(train_images, train_labels, epochs=epochs, 
                        validation_data=(test_images, test_labels), 
                        callbacks=[StreamlitCallback()])

    # Ewaluacja modelu
    st.header("Ewaluacja modelu")
    st.write("""
    Po zakończeniu trenowania modelu, dokonujemy jego ewaluacji na zbiorze testowym, aby sprawdzić, jak dobrze model radzi sobie z nowymi danymi. Testujemy model na zestawie testowym i obliczamy dokładność.
    """)
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    st.write(f'\nTest accuracy: {test_acc}')

    st.code("""
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    """, language='python')

    # Wykres dokładności
    st.header("Wykres dokładności")
    st.write("""
    Poniższy wykres przedstawia dokładność trenowania i walidacji modelu w zależności od epoki. Pozwala to zobaczyć, jak model poprawiał się w trakcie trenowania.
    """)
    fig, ax = plt.subplots()
    ax.plot(history.history['accuracy'], label='Dokładność trenowania')
    ax.plot(history.history['val_accuracy'], label='Dokładność walidacji')
    ax.set_xlabel('Epoka')
    ax.set_ylabel('Dokładność')
    ax.legend(loc='lower right')
    ax.set_title('Dokładność trenowania i walidacji')
    st.pyplot(fig)

    # Macierz konfuzji
    st.header("Macierz konfuzji")
    st.write("""
    Macierz konfuzji pozwala na ocenę, jak dobrze model klasyfikuje poszczególne klasy. Wartości na przekątnej macierzy reprezentują prawidłowe klasyfikacje, podczas gdy wartości poza przekątną reprezentują błędne klasyfikacje.
    """)
    predictions = model.predict(test_images)
    pred_labels = np.argmax(predictions, axis=1)
    cm = confusion_matrix(test_labels, pred_labels)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Macierz konfuzji')
    st.pyplot(fig)

    # Przykładowe błędnie sklasyfikowane obrazy
    st.header("Przykładowe błędnie sklasyfikowane obrazy")
    st.write("""
    Poniżej przedstawiamy przykłady obrazów, które zostały błędnie sklasyfikowane przez model. Pokazuje to, jakie rodzaje błędów model popełnia i może pomóc w dalszym ulepszaniu modelu.
    """)
    incorrect = np.where(pred_labels != test_labels)[0]
    if len(incorrect) > 0:
        fig, axes = plt.subplots(1, 5, figsize=(10, 10))
        for i, idx in enumerate(incorrect[:5]):
            axes[i].imshow(test_images[idx].reshape(28, 28), cmap='gray')
            axes[i].set_title(f'True: {test_labels[idx]}, Pred: {pred_labels[idx]}')
            axes[i].axis('off')
        st.pyplot(fig)
    else:
        st.write("Brak błędnie sklasyfikowanych obrazów.")

    # Wizualizacja filtrów konwolucyjnych
    st.header("Wizualizacja filtrów konwolucyjnych")
    st.write("""
    Filtry konwolucyjne są używane w pierwszej warstwie modelu do wykrywania podstawowych cech obrazu, takich jak krawędzie. Poniżej przedstawiamy wizualizację filtrów z pierwszej warstwy konwolucyjnej.
    """)
    first_conv_layer = model.layers[0]
    weights = first_conv_layer.get_weights()[0]

    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        if i < weights.shape[-1]:
            ax.imshow(weights[:, :, 0, i], cmap='gray')
        ax.axis('off')
    st.pyplot(fig)