import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN , KMeans 
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import re

def collect_client_data():
    print("\nPodaj dane klienta:")

    input_data = {
        'Age': clean_age(input("Wiek (np. 35): ")),
        'Annual_Income': clean_numeric(input("Roczny dochód (PLN, np. 50000): ")),
        'Monthly_Inhand_Salary': clean_numeric(input("Miesięczne wynagrodzenie netto (np. 3500): ")),
        'Num_of_Delayed_Payment': clean_numeric(input("Liczba opóźnień w płatnościach (np. 0): ")),
        'Changed_Credit_Limit': clean_numeric(input("Zmiana limitu kredytowego (np. 1000): ")),
        'Outstanding_Debt': clean_numeric(input("Zaległe zadłużenie (np. 2000): ")),
        'Amount_invested_monthly': clean_numeric(input("Kwota inwestowana miesięcznie (np. 500): ")),
        'Monthly_Balance': clean_numeric(input("Miesięczny bilans (np. 1500): ")),
        'Credit_History_Age': convert_credit_history(input("Długość historii kredytowej (np. '2 Years 3 Months'): ")),
        'Occupation': input("Zawód (np. Inżynier): ").strip(),
        'Type_of_Loan': clean_loan_types(input("Rodzaje kredytów (oddzielone przecinkami, np. 'Konsumencki, Hipoteczny'): "))
    }

    df_client = pd.DataFrame([input_data])
    df_client = process_loans(df_client)

    for col in X_train.columns:
        if col not in df_client.columns:
            if col in categorical_features:
                df_client[col] = 'missing'
            else:
                df_client[col] = np.nan


    df_client = df_client[X_train.columns]

    for col in numeric_features:
        df_client[col] = pd.to_numeric(df_client[col], errors='coerce').fillna(0)
    for col in categorical_features:
        df_client[col] = df_client[col].fillna('missing').astype(str)

    return df_client, input_data


# Funkcje pomocnicze do czyszczenia danych
def clean_numeric(value):
    try:
        return float(re.sub(r'[^0-9.]', '', str(value)))
    except:
        return np.nan

def clean_age(age):
    return abs(int(clean_numeric(age)))

def convert_credit_history(history):
    if pd.isna(history) or str(history).lower() in ['na', 'nan', '']:
        return np.nan
    try:
        years = re.search(r'(\d+)\s*Year', str(history))
        months = re.search(r'(\d+)\s*Month', str(history))
        total = 0
        if years: total += int(years.group(1)) * 12
        if months: total += int(months.group(1))
        return total if total > 0 else np.nan
    except:
        return np.nan

# Wczytanie danych z czyszczeniem
def load_and_clean_data(path):
    df = pd.read_csv(path, low_memory=False)
    
    # Czyszczenie kolumn numerycznych
    numeric_cols = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 
                'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
                'Outstanding_Debt', 'Amount_invested_monthly',
                'Monthly_Balance']
                
    for col in numeric_cols:
        df[col] = df[col].apply(clean_numeric)

    return df

train = load_and_clean_data(r'C:\Users\jakub\Visual Studio Code sem2\AWD\train.csv')
test = load_and_clean_data(r'C:\Users\jakub\Visual Studio Code sem2\AWD\test.csv')

# Przetwarzanie kolumny Age
for df in [train, test]:
    df['Age'] = df['Age'].apply(clean_age)
    df['Age'] = df.groupby('Customer_ID')['Age'].ffill().bfill()

# Konwersja historii kredytowej
for df in [train, test]:
    df['Credit_History_Age'] = df['Credit_History_Age'].apply(convert_credit_history)
    df['Credit_History_Age'] = df.groupby('Customer_ID')['Credit_History_Age'].ffill().bfill()

# Przetwarzanie zmiennej docelowej
credit_mapping = {'Poor': 20, 'Standard': 50, 'Good': 80}
train['Credit_Score'] = train['Credit_Score'].map(credit_mapping)

# Usuwanie niepotrzebnych kolumn
cols_to_drop = ['ID', 'Customer_ID', 'Month', 'Name', 'SSN']
train = train.drop(cols_to_drop, axis=1)
test = test.drop(cols_to_drop, axis=1)

# Czyszczenie kolumny Type_of_Loan
def clean_loan_types(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9, ]', '', text)
    text = re.sub(r'\s+and\s+', ', ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

for df in [train, test]:
    df['Type_of_Loan'] = df['Type_of_Loan'].apply(clean_loan_types)

# Rozbicie kolumny Type_of_Loan
def process_loans(df):
    loan_dummies = df['Type_of_Loan'].str.get_dummies(sep=', ')
    loan_dummies = loan_dummies.add_prefix('Loan_')
    df = pd.concat([df, loan_dummies], axis=1)
    return df.drop('Type_of_Loan', axis=1)

train = process_loans(train)
test = process_loans(test)

# Uzupełnianie brakujących wartości
for df in [train, test]:
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())

# Podział na cechy i target
X_train = train.drop('Credit_Score', axis=1)
y_train = train['Credit_Score']
X_test = test

# Definicja preprocessora
numeric_features = X_train.select_dtypes(include=np.number).columns
categorical_features = X_train.select_dtypes(include='object').columns

from sklearn.impute import KNNImputer

num_pipeline = Pipeline([
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, numeric_features),
        ('cat', cat_pipeline, categorical_features)
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(
        objective='reg:squarederror',
        n_estimators=200,
        max_depth=7,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ))
])

# Trenowanie i ewaluacja
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)
model.fit(X_train_split, y_train_split)
val_pred = model.predict(X_val_split)

print(f'MAE: {mean_absolute_error(y_val_split, val_pred):.2f}')
print(f'R²: {r2_score(y_val_split, val_pred):.2f}')

# Predykcja i zapis wyników
test_predictions = model.predict(X_test)
test['Predicted_Credit_Score'] = test_predictions
test[['Predicted_Credit_Score']].to_csv(r'C:\Users\jakub\Visual Studio Code sem2\AWD\credit_predictions.csv', index=False)


def menu():
    while True:
        print("\n--- MENU ---")
        print("1. Sprawdź scoring kredytowy klienta")
        print("2. Uruchom pełną analizę danych")
        print("0. Wyjście")

        choice = input("Wybierz opcję: ").strip()

        if choice == '1':
            client_df, input_data = collect_client_data()
            prediction = model.predict(client_df)[0]
            print(f"\n>>> Przewidywany scoring kredytowy klienta: {prediction:.2f}")
            if prediction >= 70:
                print("Rekomendacja: Niskie ryzyko - można zaoferować większą pożyczkę")
            elif prediction >= 50:
                print("Rekomendacja: Umiarkowane ryzyko - należy zaproponować pożyczkę warunkową")
            else:
                print("Rekomendacja: Wysokie ryzyko - pożyczka odradzana lub wymagane zabezpieczenia")

            # Uzyskaj przetworzone dane klienta (po transformacji)
            X_client_transformed = model.named_steps['preprocessor'].transform(client_df)

            # SHAP dla modelu regresji
            explainer = shap.Explainer(model.named_steps['regressor'], feature_names=model.named_steps['preprocessor'].get_feature_names_out())
            shap_values = explainer(X_client_transformed)

            # Wydobądź wartości SHAP i dopasuj nazwy
            shap_df = pd.DataFrame({
                'Cechy': shap_values.feature_names,
                'Wpływ (wartość SHAP)': shap_values.values[0]
            })

            shap_df['Cechy'] = shap_df['Cechy'].apply(lambda x: feature_name_mapping.get(x, x))
            shap_df['Wpływ (%)'] = 100 * shap_df['Wpływ (wartość SHAP)'].abs() / np.abs(shap_df['Wpływ (wartość SHAP)']).sum()
            shap_df = shap_df.sort_values(by='Wpływ (%)', ascending=False)

            # Wyodrębnij tylko te cechy, które klient faktycznie podał
            provided_feature_names = []

            # Mapujemy z input_data na pełne feature names po transformacji
            # Odwrócenie mapowania: "Wiek" -> "num__Age"
            inverse_feature_mapping = {v: k for k, v in feature_name_mapping.items()}

            # Wyciągamy tylko te cechy z SHAP, które odpowiadają danym wejściowym klienta
            provided_features = []

            for friendly_name in shap_df['Cechy']:
                internal_name = inverse_feature_mapping.get(friendly_name)
                if internal_name:
                    original_key = internal_name.replace('num__', '').replace('cat__', '')
                    if original_key in input_data:
                        provided_features.append(friendly_name)

            # Filtrowanie
            filtered_shap_df = shap_df[shap_df['Cechy'].isin(provided_features)].copy()
            filtered_shap_df = filtered_shap_df.sort_values(by='Wpływ (%)', ascending=False)

            print("\n--- Wpływ cech na scoring klienta (tylko te, które faktycznie podał): ---")
            print(filtered_shap_df.to_string(index=False))


        elif choice == '2':
            #       KLASTERYZACJA KMEANS
            # Wybór cech do klasteryzacji
            cluster_features = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Outstanding_Debt', 'Amount_invested_monthly']
            cluster_data = test[cluster_features].copy()

            # Skalowanie danych
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(cluster_data)

            # Szukanie najlepszego k (liczby klastrów)
            sil_scores = {}
            for k in range(2, 11):
                kmeans = KMeans(n_clusters=k, random_state=42)
                labels = kmeans.fit_predict(scaled_data)
                score = silhouette_score(scaled_data, labels)
                sil_scores[k] = score

            best_k = max(sil_scores, key=sil_scores.get)
            print(f'Najlepsze k = {best_k} (Silhouette score = {sil_scores[best_k]:.3f})')

            # Finalna klasteryzacja
            final_kmeans = KMeans(n_clusters=best_k, random_state=42)
            test['Cluster'] = final_kmeans.fit_predict(scaled_data)

            # Redukcja wymiarów do 2D dla wizualizacji
            pca = PCA(n_components=2)
            pca_data = pca.fit_transform(scaled_data)
            test['PCA1'] = pca_data[:, 0]
            test['PCA2'] = pca_data[:, 1]

            # Wizualizacja
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=test, x='PCA1', y='PCA2', hue='Cluster', palette='Set2')
            plt.title("Segmentacja klientów (PCA + KMeans)")
            plt.xlabel("PCA 1")
            plt.ylabel("PCA 2")
            plt.legend(title="Klaster")
            plt.tight_layout()
            plt.show()

            # Średni scoring kredytowy w każdym klastrze
            cluster_scores = test.groupby('Cluster')['Predicted_Credit_Score'].mean()
            print("\nŚredni scoring w każdej grupie klientów (klaster):")
            print(cluster_scores)

            # Rekomendacje kredytowe na podstawie klastra
            print("\nRekomendacje działań kredytowych dla każdego klastra (KMeans):")
            for cluster, score in cluster_scores.items():
                if score >= 70:
                    rekom = "Niskie ryzyko - można zaoferować większą pożyczkę"
                elif score >= 50:
                    rekom = "Umiarkowane ryzyko - należy zaproponować pożyczkę warunkową"
                else:
                    rekom = "Wysokie ryzyko - pożyczka odradzana lub wymagane zabezpieczenia"
                print(f"Klaster {cluster}: Średni scoring = {score:.2f} - {rekom}")



            print("Start DBSCAN...")

            #       KLASTERYZACJA DBSCAN
            dbscan = DBSCAN(eps=0.75, min_samples=15)
            test['DBSCAN_Cluster'] = dbscan.fit_predict(scaled_data)

            # Wizualizacja
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=test, x='PCA1', y='PCA2', hue='DBSCAN_Cluster', palette='tab10', legend='full')
            plt.title("Segmentacja klientów (PCA + DBSCAN)")
            plt.xlabel("PCA 1")
            plt.ylabel("PCA 2")
            plt.legend(title="DBSCAN Klaster")
            plt.tight_layout()
            plt.show()

            # Statystyki scoringu dla DBSCAN
            dbscan_scores = test.groupby('DBSCAN_Cluster')['Predicted_Credit_Score'].mean()
            print("\nŚredni scoring wg klastrów DBSCAN:")
            print(dbscan_scores)

            # Liczność
            print("\nLiczność klastrów DBSCAN:")
            print(test['DBSCAN_Cluster'].value_counts())




            #       ANALIZA PCA
            pca_components = pd.DataFrame(
                pca.components_, 
                columns=cluster_features, 
                index=['PCA1', 'PCA2']
            )
            print("\nWpływ cech na składowe PCA:")
            print(pca_components.T)

            #Sprawdzenie wariancji cech - mogą wpływać na "spłaszczenie" PCA 2
            variances = pd.Series(np.var(scaled_data, axis=0), index=cluster_features)
            print("\nWariancje cech użytych do klasteryzacji:")
            print(variances.sort_values(ascending=False))

            #Liczba obserwacji w klastrach
            print("\nLiczność klastrów:")
            print(test['Cluster'].value_counts())

            #Rozkład scoringu w klastrach
            plt.figure(figsize=(8, 5))
            sns.boxplot(data=test, x='Cluster', y='Predicted_Credit_Score')
            plt.title("Rozkład scoringu wg klastrów")
            plt.xlabel("Klaster")
            plt.ylabel("Przewidywany scoring kredytowy")
            plt.tight_layout()
            plt.show()




            #       FAIRNESS: ANALIZA OPISOWA
            # Fairness względem zawodu
            occupation_bias = test.groupby('Occupation')['Predicted_Credit_Score'].mean().sort_values()
            print("\nŚrednia predykcja scoringu wg zawodu:")
            print(occupation_bias)

            plt.figure(figsize=(10, 5))
            sns.barplot(x=occupation_bias.values, y=occupation_bias.index)
            plt.title("Średni scoring wg zawodu")
            plt.xlabel("Średni przewidywany scoring")
            plt.ylabel("Zawód")
            plt.tight_layout()
            plt.show()

            # Fairness względem wieku
            test['Age_Group'] = pd.cut(test['Age'], bins=[18, 25, 35, 45, 55, 70], labels=["18-25", "26-35", "36-45", "46-55", "56-70"])
            age_bias = test.groupby('Age_Group', observed=False)['Predicted_Credit_Score'].mean()
            print("\nŚrednia predykcja scoringu wg grupy wiekowej:")
            print(age_bias)

            correlation = test[['Age', 'Predicted_Credit_Score']].corr().iloc[0, 1]
            print(f"\nKorelacja wieku z przewidywanym scoringiem: {correlation:.2f}")




            #       FAIRNESS: METRYKI FORMALNE (DI & EO)
            test['Positive'] = test['Predicted_Credit_Score'] > 50

            # Wybór grupy np. wiekowej
            group_col = 'Age_Group'
            groups = test[group_col].dropna().unique()

            # Ustawienie grupy referencyjnej
            reference_group = "26-35"

            # Disparate Impact
            di_results = {}
            for group in groups:
                pos_rate_group = test[test[group_col] == group]['Positive'].mean()
                pos_rate_ref = test[test[group_col] == reference_group]['Positive'].mean()
                di = pos_rate_group / pos_rate_ref if pos_rate_ref > 0 else np.nan
                di_results[group] = di

            print("\nDISPARATE IMPACT (vs grupa referencyjna '26-35'):")
            for g, val in di_results.items():
                print(f"{g}: {val:.2f}")

            # Equal Opportunity
            y_val_bin = y_val_split > 50
            val_pred_bin = val_pred > 50

            from sklearn.metrics import confusion_matrix

            eo_groups = X_val_split.copy()
            eo_groups['y_true'] = y_val_bin.values
            eo_groups['y_pred'] = val_pred_bin
            eo_groups['Age_Group'] = pd.cut(eo_groups['Age'], bins=[18, 25, 35, 45, 55, 70],
                                            labels=["18-25", "26-35", "36-45", "46-55", "56-70"])

            print("\nEQUAL OPPORTUNITY (True Positive Rate w grupach wiekowych):")
            for group in eo_groups['Age_Group'].unique():
                sub = eo_groups[eo_groups['Age_Group'] == group]
                if sub.empty: continue
                cm = confusion_matrix(sub['y_true'], sub['y_pred'])
                TP = cm[1, 1] if cm.shape == (2, 2) else 0
                FN = cm[1, 0] if cm.shape == (2, 2) else 0
                tpr = TP / (TP + FN) if (TP + FN) > 0 else 0
                print(f"{group}: TPR = {tpr:.2f}")


        elif choice == '0':
            print("Zakończono.")
            break
        else:
            print("Nieprawidłowy wybór. Spróbuj ponownie.")

feature_name_mapping = {
    'num__Age': 'Wiek',
    'num__Annual_Income': 'Roczny dochód (PLN)',
    'num__Monthly_Inhand_Salary': 'Miesięczne wynagrodzenie netto (PLN)',
    'num__Num_Bank_Accounts': 'Liczba kont bankowych',
    'num__Num_Credit_Card': 'Liczba kart kredytowych',
    'num__Interest_Rate': 'Oprocentowanie (%)',
    'num__Num_of_Loan': 'Liczba kredytów',
    'num__Delay_from_due_date': 'Średnie opóźnienie spłaty (dni)',
    'num__Num_of_Delayed_Payment': 'Liczba opóźnionych płatności',
    'num__Changed_Credit_Limit': 'Zmiana limitu kredytowego',
    'num__Num_Credit_Inquiries': 'Liczba zapytań kredytowych',
    'cat__Credit_Mix_Bad': 'Rodzaj miksu kredytowego: Zły',
    'cat__Credit_Mix_Standard': 'Rodzaj miksu kredytowego: Standardowy',
    'cat__Credit_Mix_Good': 'Rodzaj miksu kredytowego: Dobry',
    'num__Outstanding_Debt': 'Zaległe zadłużenie (PLN)',
    'num__Credit_Utilization_Ratio': 'Wskaźnik wykorzystania kredytu (%)',
    'num__Credit_History_Age': 'Długość historii kredytowej (miesiące)',
    'cat__Payment_of_Min_Amount_Yes': 'Płatność minimalna: Tak',
    'cat__Payment_of_Min_Amount_No': 'Płatność minimalna: Nie',
    'cat__Payment_of_Min_Amount_NM': 'Płatność minimalna: Brak danych',
    'num__Total_EMI_per_month': 'Całkowita miesięczna rata (PLN)',
    'num__Amount_invested_monthly': 'Miesięczna kwota inwestycji (PLN)',
    'num__Monthly_Balance': 'Bilans miesięczny (PLN)',
}



# Wywołanie menu na końcu programu
if __name__ == "__main__":
    menu()

