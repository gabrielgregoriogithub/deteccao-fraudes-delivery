"""
Detecção de Fraudes em Delivery - Comparação de Modelos
Baseado no TCC Gabriel Gregório - PUC Minas 2022

Modelos testados (replicando e expandindo o TCC):
  - Árvore de Decisão (seção 5.2)
  - Random Forest baseline (5.2.1)
  - Random Forest + class_weight balanced (equivale ao OverSampling do TCC)
  - Random Forest + RandomizedSearchCV (5.2.2)
  - Regressão Logística (5.3)
  - XGBoost com scale_pos_weight (5.4)
  - Gradient Boosting (sklearn)
  - SVM com RBF

Métricas: Precisão, Recall, F1, AUC-ROC, Matriz de Confusão
Avaliação: train/test split 80/20 + StratifiedKFold(5) no treino
"""

import warnings
warnings.filterwarnings("ignore")
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     RandomizedSearchCV, cross_val_score)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, precision_score, recall_score,
                             f1_score, accuracy_score)
from xgboost import XGBClassifier


# ── 1. CARREGAMENTO ──────────────────────────────────────────────────────────

df = pd.read_csv("pedidos_delivery.csv")

print("=" * 65)
print("  DETECÇÃO DE FRAUDES EM DELIVERY  |  TCC Gabriel Gregório")
print("=" * 65)
print(f"\nDataset: {len(df)} pedidos")
print(f"Distribuição alvo:\n{df['fraude'].value_counts().to_string()}\n")


# ── 2. FEATURE ENGINEERING (baseado nas seções 3.1 e 4.1 do TCC) ─────────────

# hora como número inteiro (pedidos de madrugada têm maior risco)
df["hora"] = df["hora_pedido"].str.split(":").str[0].astype(int)
df["hora_risco"] = ((df["hora"] >= 22) | (df["hora"] <= 5)).astype(int)

# taxa de cancelamento (feature criada no TCC)
df["taxa_cancelamento"] = (
    df["cancelamentos_anteriores"] / df["pedidos_anteriores"].replace(0, 1)
)

# variável: cliente novo (sem histórico)
df["cliente_novo"] = (df["pedidos_anteriores"] == 0).astype(int)

# one-hot encoding das categóricas (como pd.get_dummies do TCC seção 4.1)
df = pd.get_dummies(df, columns=["dia_semana", "forma_pagamento"], drop_first=False)

# binário endereço
df["mesmo_endereco_enc"] = (df["mesmo_endereco_anterior"] == "Sim").astype(int)

# target
y = (df["fraude"] == "Sim").astype(int)

# lista de features
FEATURES = (
    ["valor_pedido", "qtd_itens", "distancia_km", "tempo_entrega_min",
     "pedidos_anteriores", "cancelamentos_anteriores", "avaliacao_anterior",
     "mesmo_endereco_enc", "taxa_cancelamento", "hora", "hora_risco", "cliente_novo"]
    + [c for c in df.columns if c.startswith("dia_semana_")]
    + [c for c in df.columns if c.startswith("forma_pagamento_")]
)

X = df[FEATURES]

print(f"Features utilizadas: {len(FEATURES)}")
print(f"  numéricas originais: valor_pedido, qtd_itens, distancia_km, "
      f"tempo_entrega_min, pedidos_anteriores, cancelamentos_anteriores, avaliacao_anterior")
print(f"  engineered: hora, hora_risco, taxa_cancelamento, cliente_novo, "
      f"mesmo_endereco_enc")
print(f"  one-hot (dia_semana + forma_pagamento): "
      f"{len([c for c in FEATURES if c.startswith(('dia_semana_','forma_pagamento_'))])} colunas")


# ── 3. SPLIT TREINO / TESTE (80/20 estratificado, como no TCC seção 5.1) ────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTreino: {len(y_train)} pedidos | Teste: {len(y_test)} pedidos")
print(f"Fraudes no treino: {y_train.sum()} ({y_train.mean():.1%})")
print(f"Fraudes no teste : {y_test.sum()} ({y_test.mean():.1%})\n")

# escalonamento para modelos lineares (Regressão Logística, SVM)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)


# ── 4. BUSCA DE HIPERPARÂMETROS - RANDOM FOREST (seção 5.2.2 do TCC) ────────

print("─" * 65)
print("Buscando melhores hiperparâmetros para Random Forest (RandomizedSearchCV)...")

param_grid_rf = {
    "n_estimators":    [50, 100, 200],
    "max_depth":       [5, 10, 15, 20, None],
    "max_features":    ["sqrt", "log2"],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
    "bootstrap":       [True, False],
    "class_weight":    [None, "balanced"],
}

rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_grid_rf,
    n_iter=30, cv=3, scoring="f1",
    random_state=42, n_jobs=-1, verbose=0
)
rf_search.fit(X_train, y_train)
best_rf = rf_search.best_params_
print(f"Melhores parâmetros RF: {best_rf}\n")


# ── 5. DEFINIÇÃO DOS MODELOS ─────────────────────────────────────────────────

# scale_pos_weight para XGBoost (equivale ao balanceamento - seção 5.4)
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
spw = round(neg / pos, 2) if pos > 0 else 1.0

MODELOS = {
    # Árvore de Decisão - seção 5.2
    "Arvore de Decisao": (
        DecisionTreeClassifier(max_depth=5, criterion="entropy", random_state=42),
        False
    ),
    # Random Forest baseline - seção 5.2.1
    "Random Forest": (
        RandomForestClassifier(n_estimators=100, max_depth=5,
                               bootstrap=False, random_state=42),
        False
    ),
    # RF com balanceamento (equivale ao OverSampling do TCC)
    "Random Forest Balanceado": (
        RandomForestClassifier(n_estimators=100, class_weight="balanced",
                               random_state=42),
        False
    ),
    # RF tunado com RandomizedSearchCV - seção 5.2.2
    "Random Forest Tunado": (
        RandomForestClassifier(**best_rf, random_state=42),
        False
    ),
    # Regressão Logística - seção 5.3
    "Regressao Logistica": (
        LogisticRegression(max_iter=500, class_weight="balanced",
                           solver="lbfgs", random_state=42),
        True   # requer escalonamento
    ),
    # XGBoost - seção 5.4
    "XGBoost": (
        XGBClassifier(
            max_depth=9,
            scale_pos_weight=spw,
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=100,
            random_state=42,
            verbosity=0,
        ),
        False
    ),
    # Gradient Boosting (sklearn) - modelo adicional
    "Gradient Boosting": (
        GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                   random_state=42),
        False
    ),
    # SVM com RBF - modelo adicional
    "SVM": (
        SVC(kernel="rbf", class_weight="balanced",
            probability=True, random_state=42),
        True   # requer escalonamento
    ),
}


# ── 6. AVALIAÇÃO DE TODOS OS MODELOS ─────────────────────────────────────────

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
resultados = []

print("=" * 65)
print("RESULTADOS POR MODELO")
print("=" * 65)

for nome, (modelo, usar_scaler) in MODELOS.items():
    Xtr = X_train_s if usar_scaler else X_train.values
    Xte = X_test_s  if usar_scaler else X_test.values

    # cross-validation no conjunto de treino (5 folds)
    cv_prec = cross_val_score(modelo, Xtr, y_train, cv=skf,
                              scoring="precision").mean()
    cv_rec  = cross_val_score(modelo, Xtr, y_train, cv=skf,
                              scoring="recall").mean()
    cv_f1   = cross_val_score(modelo, Xtr, y_train, cv=skf,
                              scoring="f1").mean()

    # treino e predição no teste
    modelo.fit(Xtr, y_train)
    y_pred = modelo.predict(Xte)
    y_prob = modelo.predict_proba(Xte)[:, 1]

    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    acc  = accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)
    cm   = confusion_matrix(y_test, y_pred)

    resultados.append({
        "Modelo":      nome,
        "Precisao":    prec,
        "Recall":      rec,
        "F1":          f1,
        "Acuracia":    acc,
        "AUC-ROC":     auc,
        "CV_Precisao": cv_prec,
        "CV_Recall":   cv_rec,
        "CV_F1":       cv_f1,
        "TP": int(cm[1, 1]),
        "FP": int(cm[0, 1]),
        "FN": int(cm[1, 0]),
        "TN": int(cm[0, 0]),
    })

    print(f"\n{'─'*65}")
    print(f"  {nome.upper()}")
    print(f"{'─'*65}")
    print(f"  Matriz de Confusão (teste):")
    print(f"                   Previsto Legítimo  Previsto Fraude")
    print(f"  Real Legítimo         {cm[0,0]:>5}             {cm[0,1]:>5}")
    print(f"  Real Fraude           {cm[1,0]:>5}             {cm[1,1]:>5}")
    print(f"\n  TESTE  → Precisão={prec:.2%} | Recall={rec:.2%} | "
          f"F1={f1:.2%} | AUC={auc:.4f} | Acurácia={acc:.2%}")
    print(f"  CV(5)  → Precisão={cv_prec:.2%} | Recall={cv_rec:.2%} | F1={cv_f1:.2%}")
    print()
    print(classification_report(y_test, y_pred,
                                 target_names=["Legitimo", "Fraude"],
                                 zero_division=0))


# ── 7. TABELA COMPARATIVA FINAL (como seção 6 do TCC) ───────────────────────

df_res = pd.DataFrame(resultados).sort_values("Precisao", ascending=False)

print("\n" + "=" * 65)
print("TABELA COMPARATIVA FINAL - ordenada por Precisão (classe Fraude)")
print("=" * 65)
print(f"\n{'Modelo':<28} {'Precisao':>9} {'Recall':>8} {'F1':>7} "
      f"{'AUC-ROC':>8} {'TP':>4} {'FP':>4} {'FN':>4}")
print("─" * 65)
for _, r in df_res.iterrows():
    print(f"  {r['Modelo']:<26} {r['Precisao']:>8.2%} {r['Recall']:>7.2%} "
          f"{r['F1']:>6.2%} {r['AUC-ROC']:>8.4f} "
          f"{r['TP']:>4} {r['FP']:>4} {r['FN']:>4}")

print("\n  TP=Fraudes bloqueadas corretamente | FP=Legítimos bloqueados indevido")
print("  FN=Fraudes que passaram | Precisão=métrica principal (TCC seção 5.1)")


# ── 8. CROSS-VALIDATION COMPARATIVO (mais estável p/ datasets pequenos) ──────

print("\n" + "=" * 65)
print("CROSS-VALIDATION 5-FOLD NO TREINO (estimativa mais estável)")
print("=" * 65)
print(f"\n{'Modelo':<28} {'CV-Precisao':>12} {'CV-Recall':>10} {'CV-F1':>8}")
print("─" * 65)
df_cv = df_res.sort_values("CV_F1", ascending=False)
for _, r in df_cv.iterrows():
    print(f"  {r['Modelo']:<26} {r['CV_Precisao']:>11.2%} "
          f"{r['CV_Recall']:>9.2%} {r['CV_F1']:>7.2%}")


# ── 9. IMPORTÂNCIA DE VARIÁVEIS (modelos baseados em árvore) ─────────────────

print("\n" + "=" * 65)
print("IMPORTÂNCIA DE VARIÁVEIS - TOP 15")
print("=" * 65)

modelos_importancia = {
    k: v[0] for k, v in MODELOS.items()
    if hasattr(v[0], "feature_importances_")
}

for nome, modelo in modelos_importancia.items():
    imp = pd.Series(modelo.feature_importances_, index=FEATURES)
    imp = imp[imp > 0].sort_values(ascending=False).head(15)
    print(f"\n  {nome}:")
    for feat, val in imp.items():
        bar = "█" * max(1, int(val * 150))
        print(f"    {feat:<35} {val:.4f}  {bar}")


# ── 10. MELHOR MODELO (por Precisão - critério principal do TCC) ──────────────

melhor = df_res.iloc[0]
melhor_f1 = df_res.sort_values("F1", ascending=False).iloc[0]
melhor_auc = df_res.sort_values("AUC-ROC", ascending=False).iloc[0]

print(f"\n{'='*65}")
print("CONCLUSÃO")
print(f"{'='*65}")
print(f"\n  Melhor Precisão : {melhor['Modelo']:<30} "
      f"Precisão={melhor['Precisao']:.2%} | Recall={melhor['Recall']:.2%}")
print(f"  Melhor F1-Score : {melhor_f1['Modelo']:<30} "
      f"F1={melhor_f1['F1']:.2%}")
print(f"  Melhor AUC-ROC  : {melhor_auc['Modelo']:<30} "
      f"AUC={melhor_auc['AUC-ROC']:.4f}")

print(f"\n  Referência TCC (mercado): 95% aprovação, 0.7% CBK")
print(f"  Modelo selecionado '{melhor['Modelo']}':")
total_fraudes = melhor["TP"] + melhor["FN"]
total_legitimos = melhor["TN"] + melhor["FP"]
print(f"    Fraudes capturadas : {melhor['TP']}/{total_fraudes} ({melhor['TP']/max(total_fraudes,1):.0%})")
print(f"    Legítimos aprovados: {melhor['TN']}/{total_legitimos} ({melhor['TN']/max(total_legitimos,1):.0%})")
print()
