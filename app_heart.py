import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              confusion_matrix, roc_curve)
from sklearn.linear_model  import LogisticRegression
from sklearn.neighbors      import KNeighborsClassifier
from sklearn.tree           import DecisionTreeClassifier
from sklearn.ensemble       import RandomForestClassifier
from sklearn.svm            import SVC
from sklearn.naive_bayes    import GaussianNB
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #e74c3c;
        margin-bottom: 0;
    }
    .sub-title {
        color: #7f8c8d;
        font-size: 1rem;
        margin-top: 0;
    }
    .best-badge {
        background: linear-gradient(135deg, #f39c12, #e67e22);
        color: white;
        border-radius: 8px;
        padding: 6px 14px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
        margin-left: 8px;
    }
    .risk-high {
        background: linear-gradient(135deg, #c0392b, #e74c3c);
        color: white;
        border-radius: 12px;
        padding: 20px;
        font-size: 1.4rem;
        font-weight: bold;
        text-align: center;
    }
    .risk-low {
        background: linear-gradient(135deg, #1e8449, #2ecc71);
        color: white;
        border-radius: 12px;
        padding: 20px;
        font-size: 1.4rem;
        font-weight: bold;
        text-align: center;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #2c3e50;
        border-bottom: 2px solid #e74c3c;
        padding-bottom: 5px;
        margin-bottom: 15px;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    div[data-testid="stSidebar"] * {
        color: #ecf0f1 !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ALL MODELS MAP
# ─────────────────────────────────────────────
MODELS_MAP = {
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM":                 SVC(probability=True, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "Naive Bayes":         GaussianNB(),
}

# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    df = df.copy()
    df["Sex"]            = df["Sex"].map({"M": 1, "F": 0})
    df["ExerciseAngina"] = df["ExerciseAngina"].map({"Y": 1, "N": 0})
    df = pd.get_dummies(df,
                        columns=["ChestPainType", "RestingECG", "ST_Slope"],
                        drop_first=True)
    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]
    for col in ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]:
        Q1, Q3 = X[col].quantile(0.25), X[col].quantile(0.75)
        X[col] = X[col].clip(Q1 - 1.5 * (Q3 - Q1), Q3 + 1.5 * (Q3 - Q1))
    return X, y

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<p class="main-title">🫀 Heart Disease Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">تنبؤ بأمراض القلب باستخدام نماذج Machine Learning المتقدمة</p>', unsafe_allow_html=True)
st.divider()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📂 رفع الداتا")
    uploaded = st.file_uploader("ارفع ملف heart.csv", type=["csv"])

    st.markdown("---")
    st.markdown("## 🤖 اختيار النموذج")
    model_name = st.selectbox(
        "النموذج",
        list(MODELS_MAP.keys()),
        help="Random Forest عادةً بيحقق أعلى accuracy"
    )

    st.markdown("---")
    show_compare = st.checkbox("📊 قارن كل النماذج", value=False)

    st.markdown("---")
    st.markdown("## ℹ️ معلومات")
    st.info("الداتا: 918 مريض | 12 feature\nTarget: HeartDisease (0/1)\nModels: 6 نماذج sklearn")

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data
def make_demo_data():
    np.random.seed(42)
    n = 918
    hd = np.random.binomial(1, 0.55, n)
    age = np.where(hd, np.random.normal(58, 9, n), np.random.normal(52, 10, n)).clip(28, 77).astype(int)
    max_hr = np.where(hd, np.random.normal(130, 25, n), np.random.normal(155, 22, n)).clip(60, 202).astype(int)
    oldpeak = np.where(hd, np.random.exponential(1.5, n), np.random.exponential(0.5, n)).clip(0, 6).round(1)
    return pd.DataFrame({
        "Age":            age,
        "Sex":            np.random.choice(["M", "F"], n, p=[0.75, 0.25]),
        "ChestPainType":  np.where(hd,
                              np.random.choice(["ASY","NAP","ATA","TA"], n, p=[0.6,0.2,0.15,0.05]),
                              np.random.choice(["ATA","NAP","ASY","TA"], n, p=[0.5,0.3,0.15,0.05])),
        "RestingBP":      np.random.randint(80, 200, n),
        "Cholesterol":    np.random.randint(100, 400, n),
        "FastingBS":      np.random.binomial(1, 0.23, n),
        "RestingECG":     np.random.choice(["Normal","ST","LVH"], n, p=[0.59,0.19,0.22]),
        "MaxHR":          max_hr,
        "ExerciseAngina": np.where(hd,
                              np.random.choice(["Y","N"], n, p=[0.65,0.35]),
                              np.random.choice(["Y","N"], n, p=[0.23,0.77])),
        "Oldpeak":        oldpeak,
        "ST_Slope":       np.where(hd,
                              np.random.choice(["Flat","Down","Up"], n, p=[0.55,0.25,0.20]),
                              np.random.choice(["Up","Flat","Down"],  n, p=[0.65,0.29,0.06])),
        "HeartDisease":   hd,
    })

if uploaded:
    raw_df = pd.read_csv(uploaded)
    st.success(f"✅ تم تحميل الملف: {raw_df.shape[0]} صف × {raw_df.shape[1]} عمود")
else:
    raw_df = make_demo_data()
    st.warning("⚠️ لم يتم رفع ملف — يتم استخدام بيانات تجريبية. ارفع heart.csv من الـ Sidebar.")

# ─────────────────────────────────────────────
# PREPROCESS & SPLIT & SCALE
# ─────────────────────────────────────────────
@st.cache_data
def prepare(df_hash):
    X, y = preprocess(df_hash)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    sc = StandardScaler()
    X_tr_s = pd.DataFrame(sc.fit_transform(X_tr), columns=X.columns)
    X_te_s  = pd.DataFrame(sc.transform(X_te),    columns=X.columns)
    return X, y, X_tr_s, X_te_s, y_tr, y_te, sc

X, y, X_tr_s, X_te_s, y_tr, y_te, sc = prepare(raw_df)

# ─────────────────────────────────────────────
# TRAIN SINGLE MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def train_model(name, _X_tr, _y_tr):
    import copy
    m = copy.deepcopy(MODELS_MAP[name])
    m.fit(_X_tr, _y_tr)
    return m

with st.spinner(f"⏳ جاري تدريب نموذج {model_name}..."):
    model = train_model(model_name, X_tr_s, y_tr)

y_pred = model.predict(X_te_s)
y_prob = model.predict_proba(X_te_s)[:, 1]

acc  = accuracy_score(y_te, y_pred)
f1   = f1_score(y_te, y_pred)
auc_ = roc_auc_score(y_te, y_prob)
cm   = confusion_matrix(y_te, y_pred)

# ─────────────────────────────────────────────
# COMPARE ALL MODELS (optional)
# ─────────────────────────────────────────────
@st.cache_data
def compare_all(_X_tr, _X_te, _y_tr, _y_te):
    import copy
    rows = []
    for name, clf in MODELS_MAP.items():
        m = copy.deepcopy(clf)
        m.fit(_X_tr, _y_tr)
        yp = m.predict(_X_te)
        ypr = m.predict_proba(_X_te)[:, 1]
        rows.append({
            "النموذج": name,
            "Accuracy": accuracy_score(_y_te, yp),
            "F1 Score": f1_score(_y_te, yp),
            "ROC-AUC":  roc_auc_score(_y_te, ypr),
        })
    df_r = pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    df_r.index += 1
    return df_r

if show_compare:
    with st.spinner("⏳ جاري تدريب كل النماذج للمقارنة..."):
        df_compare = compare_all(X_tr_s, X_te_s, y_tr, y_te)

    best_model_name = df_compare.iloc[0]["النموذج"]
    best_acc        = df_compare.iloc[0]["Accuracy"]

    st.success(f"🏆 أعلى Accuracy: **{best_model_name}** — `{best_acc:.2%}`")

    styled = df_compare.style\
        .background_gradient(subset=["Accuracy","F1 Score","ROC-AUC"], cmap="Greens")\
        .format({"Accuracy":"{:.4f}","F1 Score":"{:.4f}","ROC-AUC":"{:.4f}"})\
        .set_caption("📊 مقارنة كل النماذج — مرتبة حسب Accuracy")
    st.dataframe(styled, use_container_width=True)

    # Bar chart
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#e74c3c" if n == best_model_name else "#3498db"
              for n in df_compare["النموذج"]]
    bars = ax.barh(df_compare["النموذج"][::-1], df_compare["Accuracy"][::-1],
                   color=colors[::-1], edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, df_compare["Accuracy"][::-1]):
        ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=10, fontweight="bold")
    ax.set_xlim(0.5, 1.05)
    ax.axvline(df_compare["Accuracy"].mean(), color="orange",
               linestyle="--", label=f'Mean = {df_compare["Accuracy"].mean():.4f}')
    ax.set_title("Accuracy Comparison — All Models", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.divider()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 أداء النموذج", "🔍 تنبؤ مريض جديد", "📈 استكشاف الداتا"])

# ═══════════════════════════════════════════
# TAB 1 — MODEL PERFORMANCE
# ═══════════════════════════════════════════
with tab1:
    st.markdown(f'<p class="section-header">نتائج نموذج {model_name}</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎯 Accuracy",   f"{acc:.2%}")
    c2.metric("⚖️ F1 Score",   f"{f1:.4f}")
    c3.metric("📉 ROC-AUC",    f"{auc_:.4f}")
    tn, fp, fn, tp = cm.ravel()
    c4.metric("✅ Specificity", f"{tn/(tn+fp):.2%}")

    st.divider()
    col_a, col_b = st.columns(2)

    # Confusion Matrix
    with col_a:
        st.markdown("**Confusion Matrix**")
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["No Disease", "Disease"], fontsize=11)
        ax.set_yticklabels(["No Disease", "Disease"], fontsize=11)
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual",    fontsize=12)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=18, fontweight="bold",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ROC Curve
    with col_b:
        st.markdown("**ROC Curve**")
        fpr, tpr, _ = roc_curve(y_te, y_prob)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, color="#e74c3c", linewidth=2.5,
                label=f"AUC = {auc_:.4f}")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Random")
        ax.fill_between(fpr, tpr, alpha=0.1, color="#e74c3c")
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate",  fontsize=12)
        ax.set_title(f"ROC Curve — {model_name}", fontsize=13, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Feature Importance
    st.divider()
    st.markdown("**🔑 Feature Importance**")
    if hasattr(model, "feature_importances_"):
        fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#e74c3c" if v >= fi.quantile(0.75) else "#3498db" for v in fi.values]
        ax.barh(fi.index, fi.values, color=colors, edgecolor="white", linewidth=1)
        ax.set_xlabel("Importance", fontsize=12)
        ax.set_title("Feature Importance", fontsize=13, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    elif hasattr(model, "coef_"):
        coefs = pd.Series(np.abs(model.coef_[0]), index=X.columns).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(coefs.index, coefs.values, color="#3498db", edgecolor="white", linewidth=1)
        ax.set_xlabel("|Coefficient|", fontsize=12)
        ax.set_title("Feature Coefficients (Logistic Regression)", fontsize=13, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.info("Feature importance غير متاح مباشرةً لهذا النموذج.")

# ═══════════════════════════════════════════
# TAB 2 — PATIENT PREDICTION
# ═══════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-header">🔍 إدخال بيانات مريض جديد</p>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**المعلومات الأساسية**")
        age        = st.slider("العمر (Age)", 18, 80, 54)
        sex        = st.selectbox("الجنس (Sex)", ["ذكر (Male)", "أنثى (Female)"])
        resting_bp = st.slider("ضغط الدم (RestingBP)", 80, 200, 130)
        cholest    = st.slider("الكوليسترول (Cholesterol)", 100, 400, 200)

    with c2:
        st.markdown("**نتائج مخطط القلب**")
        fasting_bs = st.selectbox("سكر الصيام > 120 (FastingBS)", [0, 1])
        max_hr     = st.slider("أقصى معدل قلب (MaxHR)", 60, 210, 150)
        oldpeak    = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0, step=0.1)
        ex_angina  = st.selectbox("ذبحة صدرية بالمجهود", ["لا (No)", "نعم (Yes)"])

    with c3:
        st.markdown("**التصنيفات**")
        chest_pain  = st.selectbox("نوع ألم الصدر (ChestPainType)", ["ATA", "NAP", "ASY", "TA"])
        resting_ecg = st.selectbox("تخطيط القلب (RestingECG)", ["Normal", "ST", "LVH"])
        st_slope    = st.selectbox("ميل ST (ST_Slope)", ["Up", "Flat", "Down"])

    st.divider()
    if st.button("🫀 تنبؤ بوجود أمراض القلب", use_container_width=True, type="primary"):
        sex_v   = 1 if "Male" in sex else 0
        ex_v    = 1 if "Yes" in ex_angina else 0
        cp_nap  = int(chest_pain == "NAP")
        cp_ta   = int(chest_pain == "TA")
        cp_asy  = int(chest_pain == "ASY")
        ecg_st  = int(resting_ecg == "ST")
        ecg_lvh = int(resting_ecg == "LVH")
        sl_fl   = int(st_slope == "Flat")
        sl_up   = int(st_slope == "Up")

        row = pd.DataFrame([[age, sex_v, resting_bp, cholest, fasting_bs,
                             max_hr, ex_v, oldpeak,
                             cp_nap, cp_ta, cp_asy,
                             ecg_st, ecg_lvh,
                             sl_fl, sl_up]],
                           columns=X.columns)
        row_sc = sc.transform(row)
        pred = model.predict(row_sc)[0]
        prob = model.predict_proba(row_sc)[0][1]

        if pred == 1:
            st.markdown(
                f'<div class="risk-high">🔴 خطر مرتفع لأمراض القلب!&nbsp;&nbsp;&nbsp;'
                f'الاحتمالية: {prob:.1%}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="risk-low">🟢 خطر منخفض لأمراض القلب&nbsp;&nbsp;&nbsp;'
                f'الاحتمالية: {prob:.1%}</div>',
                unsafe_allow_html=True,
            )

        st.progress(float(prob))

        # Risk gauge
        st.divider()
        col_g, col_s = st.columns([1, 2])
        with col_g:
            fig, ax = plt.subplots(figsize=(4, 3), subplot_kw={"projection": "polar"})
            theta = np.linspace(0, np.pi, 300)
            ax.plot(theta, [1] * 300, color="lightgray", linewidth=20, solid_capstyle="round")
            theta_val = np.linspace(0, prob * np.pi, 300)
            color = "#e74c3c" if prob > 0.5 else "#2ecc71"
            ax.plot(theta_val, [1] * len(theta_val), color=color, linewidth=20, solid_capstyle="round")
            ax.set_ylim(0, 1.5)
            ax.set_theta_zero_location("W")
            ax.set_theta_direction(1)
            ax.axis("off")
            ax.text(np.pi / 2, 0.3, f"{prob:.1%}", ha="center", va="center",
                    fontsize=18, fontweight="bold", color=color)
            ax.set_title("نسبة الخطر", fontsize=12, fontweight="bold", pad=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_s:
            st.markdown("**ملخص القيم المُدخَلة:**")
            summary = pd.DataFrame({
                "الخاصية":  ["العمر", "الجنس", "ضغط الدم", "الكوليسترول",
                              "أقصى معدل قلب", "Oldpeak", "نوع ألم الصدر", "ST_Slope"],
                "القيمة":   [age, sex, resting_bp, cholest,
                             max_hr, oldpeak, chest_pain, st_slope],
            })
            st.dataframe(summary, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════
# TAB 3 — DATA EXPLORATION
# ═══════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-header">📈 استكشاف الداتا</p>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**توزيع الهدف (HeartDisease)**")
        counts = raw_df["HeartDisease"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(["No Disease (0)", "Disease (1)"], counts.values,
                      color=["#2ecc71", "#e74c3c"], edgecolor="white", linewidth=2)
        for bar, cnt in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    f"{cnt}\n({cnt/len(raw_df):.1%})",
                    ha="center", fontsize=12, fontweight="bold")
        ax.set_ylim(0, max(counts.values) * 1.2)
        ax.set_title("Heart Disease Distribution", fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown("**توزيع العمر حسب الحالة**")
        fig, ax = plt.subplots(figsize=(5, 4))
        for val, color, label in zip([0, 1], ["#2ecc71", "#e74c3c"], ["No Disease", "Disease"]):
            ax.hist(raw_df[raw_df["HeartDisease"] == val]["Age"], bins=20,
                    alpha=0.65, color=color, label=label, edgecolor="white")
        ax.set_xlabel("Age", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Age Distribution by Heart Disease", fontsize=13, fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.divider()
    st.markdown("**الخصائص الفئوية مقابل أمراض القلب**")
    cat_col = st.selectbox("اختر خاصية",
                           ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"])
    ct = pd.crosstab(raw_df[cat_col], raw_df["HeartDisease"], normalize="index") * 100
    fig, ax = plt.subplots(figsize=(8, 4))
    ct.plot(kind="bar", ax=ax, color=["#2ecc71", "#e74c3c"],
            edgecolor="white", linewidth=1.5, rot=0)
    ax.set_ylabel("النسبة المئوية (%)")
    ax.set_title(f"{cat_col} vs HeartDisease", fontsize=13, fontweight="bold")
    ax.legend(["No Disease", "Disease"])
    ax.set_ylim(0, 115)
    for bar in ax.patches:
        h = bar.get_height()
        if h > 3:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                    f"{h:.0f}%", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.divider()
    st.markdown("**📋 عينة من الداتا**")
    st.dataframe(raw_df.head(20), use_container_width=True)

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("**إحصاءات وصفية**")
        st.dataframe(raw_df.describe().round(2), use_container_width=True)
    with col_s2:
        st.markdown("**القيم المفقودة**")
        missing = raw_df.isnull().sum().reset_index()
        missing.columns = ["Feature", "Missing Count"]
        st.dataframe(missing, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.divider()
st.markdown(
    "<center><small>🫀 Heart Disease Predictor — مبني على Heart Failure Prediction Dataset (918 مريض) | "
    "Models: Random Forest · SVM · Logistic Regression · KNN · Decision Tree · Naive Bayes</small></center>",
    unsafe_allow_html=True,
)
