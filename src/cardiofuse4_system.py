import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
import xgboost as xgb

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


# =====================================================================
# 1. MULTI-MODAL SYNTHETIC DATA GENERATOR (Matching Papers 1, 2, & 3)
# =====================================================================
def generate_multimodal_cohort(n_samples=1200):
    """
    Simulates a unified patient cohort containing:
    - Modality 1: 19 Lifestyle Pivotal Factors (Paper 3)
    - Modality 2: 87 Clinical/Lab Attributes -> filtered to 26 (Paper 1)
    - Modality 3: 4 Categorical + 3 Clinical Text EMR fields (Paper 2)
    - Modality 4: Raw ECG Time-Series Signals (Paper 2)
    """
    # Ground-truth latent cardiovascular risk
    latent_risk = np.random.normal(0, 1, n_samples)
    y = (latent_risk > 0.15).astype(int)  # Imbalanced binary CVD label

    # --- Modality 1: 19 Individual Pivotal / Lifestyle Factors (Paper 3) ---
    lifestyle_cols = [
        "BMI", "Weight_kg", "Age_min", "FriedPotato_Consumption",
        "Green_Vegetables_Consumption", "Fruit_Consumption",
        "Alcohol_Consumption", "General_Health", "Height_cm", "Diabetes",
        "Sex", "Checkup", "Smoking_History", "Arthritis", "Other_Cancer",
        "Depression", "Skin_Cancer", "Exercise", "Age_max"
    ]
    X_lifestyle = np.random.randn(n_samples, 19)
    # Inject strong signal into top contributors (BMI, Weight, Age)
    X_lifestyle[:, 0] += latent_risk * 1.1  # BMI
    X_lifestyle[:, 1] += latent_risk * 0.9  # Weight
    X_lifestyle[:, 2] += latent_risk * 0.8  # Age_min
    df_lifestyle = pd.DataFrame(X_lifestyle, columns=lifestyle_cols)

    # --- Modality 2: 87 Lab/Biomarker Attributes (Paper 1) ---
    named_26 = [
        "Age", "Gender", "Systolic", "Diastolic", "BMI", "WBC", "Lymph",
        "TG", "TChol", "Hemoglobin", "Baso", "MCV", "MCHC", "PC", "ALB",
        "AST", "ALT", "ALP", "Gluc", "P", "Protein_Chest_Tightness",
        "HDL", "Moderate_work", "Diabetes", "Blood_Rel_Diabetes",
        "Blood_Rel_Stroke"
    ]
    noise_cols = [f"Noise_Lab_{i}" for i in range(87 - 26)]
    X_lab_core = np.random.randn(n_samples, 26) + (latent_risk[:, None] * 0.45)
    X_lab_noise = np.random.randn(n_samples, 87 - 26) * 1.5
    df_labs = pd.DataFrame(
        np.hstack([X_lab_core, X_lab_noise]),
        columns=named_26 + noise_cols
    )

    # --- Modality 3: EMR Data (4 Categorical + 3 Textual) (Paper 2) ---
    df_emr = pd.DataFrame({
        "gender": np.random.choice(["Male", "Female"], n_samples),
        "age_bin": np.random.choice(["<30", "30-50", "50-70", "70-89"], n_samples),
        "race": np.random.choice(["GroupA", "GroupB", "GroupC"], n_samples),
        "ethnicity": np.random.choice(["Hispanic", "Nonhispanic", "Unreported"], n_samples),
    })
    high_risk_terms = ["chest pain dyspnea", "coronary ischemia st elevation", "tachycardia angina"]
    low_risk_terms = ["routine checkup", "mild fatigue normal sinus", "annual physical stable"]

    reasons, diagnoses, orders = [], [], []
    for label in y:
        if label == 1 and np.random.rand() < 0.75:
            reasons.append(np.random.choice(high_risk_terms))
            diagnoses.append("suspected ischemic heart disease i20")
            orders.append("cardiac troponin ecg monitoring")
        else:
            reasons.append(np.random.choice(low_risk_terms))
            diagnoses.append("general observation")
            orders.append("standard vitals discharge")
    df_emr["reason_for_admission"] = reasons
    df_emr["diagnosis"] = diagnoses
    df_emr["admit_order"] = orders

    # --- Modality 4: Analog ECG Segments (Paper 2) ---
    # Using D=200 time steps per segment for fast CPU execution (scalable to 6000 for 6s @ 1kHz)
    D_time = 200
    t = np.linspace(0, 6 * np.pi, D_time)
    X_ecg = np.zeros((n_samples, D_time))
    for i in range(n_samples):
        freq_shift = 1.0 + (0.35 * y[i])
        st_depression = -0.4 * y[i] * np.sin(2 * t)
        X_ecg[i] = np.sin(freq_shift * t) + st_depression + np.random.normal(0, 0.25, D_time)

    return df_lifestyle, df_labs, df_emr, X_ecg, y


# =====================================================================
# 2. STREAM 1: XGBOOST FEATURE SELECTION + LOCAL CASCADE ENSEMBLE (Paper 3)
# =====================================================================
class LocalCascadeEnsembleClassifier:
    """
    Implements the Local Cascade Ensemble (LCE) principle from Paper 3:
    Combines Boosting (XGBoost base learners adding class probabilities across levels)
    and Bagging (Random Forest sub-trees / Mixture of Experts) for high generalization.
    """
    def __init__(self, n_Select=15):
        self.n_select = n_Select
        self.selected_features = None
        self.feature_selector = xgb.XGBClassifier(
            n_estimators=50, max_depth=4, eval_metric="logloss", random_state=42
        )
        # Level 0 Boosted Base Learner (H_b)
        self.level0_xgb = xgb.XGBClassifier(
            n_estimators=80, max_depth=4, learning_rate=0.08, eval_metric="logloss", random_state=42
        )
        # Level 1 Bagging + Boosting Cascade Learners on probability-augmented dataset D_1
        self.cascade_rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        self.cascade_xgb = xgb.XGBClassifier(
            n_estimators=80, max_depth=3, learning_rate=0.05, eval_metric="logloss", random_state=42
        )

    @staticmethod
    def apply_smote_2_to_3(X, y):
        """Rebalances positive-to-negative class ratio toward 2:3 using interpolation."""
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        target_pos = int(len(neg_idx) * (2.0 / 3.0))
        if len(pos_idx) >= target_pos:
            return X, y
        n_synth = target_pos - len(pos_idx)
        X_pos = X[pos_idx]
        synth_samples = []
        for _ in range(n_synth):
            i, j = np.random.choice(len(X_pos), 2, replace=True)
            lam = np.random.uniform(0, 1)
            synth_samples.append(X_pos[i] + lam * (X_pos[j] - X_pos[i]))
        X_balanced = np.vstack([X, np.array(synth_samples)])
        y_balanced = np.concatenate([y, np.ones(n_synth, dtype=int)])
        return X_balanced, y_balanced

    def fit(self, df_X, y):
        # Step 1: XGBoost Feature Selection (Top 15 pivotal features)
        self.feature_selector.fit(df_X, y)
        importances = self.feature_selector.feature_importances_
        top_idx = np.argsort(importances)[::-1][:self.n_select]
        self.selected_features = df_X.columns[top_idx].tolist()

        X_sel = df_X[self.selected_features].values
        X_bal, y_bal = self.apply_smote_2_to_3(X_sel, y)

        # Step 2: Level 0 Base Boosting (D_0 = D + H_b(D))
        self.level0_xgb.fit(X_bal, y_bal)
        prob_level0 = self.level0_xgb.predict_proba(X_bal)
        X_augmented = np.hstack([X_bal, prob_level0])

        # Step 3: Level 1 Cascade Mixture of Experts
        self.cascade_rf.fit(X_augmented, y_bal)
        self.cascade_xgb.fit(X_augmented, y_bal)
        return self

    def predict_proba(self, df_X):
        X_sel = df_X[self.selected_features].values
        prob_level0 = self.level0_xgb.predict_proba(X_sel)
        X_augmented = np.hstack([X_sel, prob_level0])
        p_rf = self.cascade_rf.predict_proba(X_augmented)[:, 1]
        p_xgb = self.cascade_xgb.predict_proba(X_augmented)[:, 1]
        return 0.5 * (p_rf + p_xgb)


# =====================================================================
# 3. STREAM 2: 1D-CNN + SELF-ATTENTION BIOMARKER MODEL (Paper 1)
# =====================================================================
class DeepFeature1DCNNAttention(nn.Module):
    """
    Implements Paper 1: Improved 1D-CNN + Self-Attention + Softmax.
    """
    def __init__(self, num_features=26):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # Self-Attention over extracted feature channels
        self.attention = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)
        seq_len_after_pool = num_features // 2
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * seq_len_after_pool, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x shape: (batch, num_features) -> (batch, 1, num_features)
        x = x.unsqueeze(1)
        conv_out = self.conv_block(x)          # (batch, 32, seq_len)
        seq = conv_out.permute(0, 2, 1)        # (batch, seq_len, 32)
        attn_out, _ = self.attention(seq, seq, seq)
        probs = self.classifier(attn_out)
        return probs


class BiomarkerDeepPipeline:
    """Handles Paper 1 RF Gini Selection (87 -> 30) + Pearson Filtering (30 -> 26) + 1D-CNN Training."""
    def __init__(self, n_rf_keep=30, n_final_keep=26):
        self.n_rf_keep = n_rf_keep
        self.n_final_keep = n_final_keep
        self.selected_cols = None
        self.scaler = StandardScaler()
        self.model = DeepFeature1DCNNAttention(num_features=n_final_keep)

    def select_features(self, df_X, y):
        # 1. Random Forest Gini Impurity Ranking (87 -> 30)
        rf = RandomForestClassifier(n_estimators=100, criterion="gini", random_state=42)
        rf.fit(df_X, y)
        top30_idx = np.argsort(rf.feature_importances_)[::-1][:self.n_rf_keep]
        top30_cols = df_X.columns[top30_idx]

        # 2. Pearson Correlation Ranking with Target (30 -> 26)
        corrs = df_X[top30_cols].apply(lambda col: abs(np.corrcoef(col, y)[0, 1]))
        self.selected_cols = corrs.sort_values(ascending=False).index[:self.n_final_keep].tolist()

    def fit(self, df_X, y, epochs=35, lr=0.003):
        self.select_features(df_X, y)
        X_scaled = self.scaler.fit_transform(df_X[self.selected_cols].values)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            probs = self.model(X_tensor)
            loss = criterion(probs, y_tensor)
            loss.backward()
            optimizer.step()
        return self

    def predict_proba(self, df_X):
        self.model.eval()
        X_scaled = self.scaler.transform(df_X[self.selected_cols].values)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            probs = self.model(X_tensor)[:, 1].numpy()
        return probs


# =====================================================================
# 4. STREAM 3: EMR TF-IDF + LINEAR SVM PIPELINE (Paper 2)
# =====================================================================
class EMRPipeline:
    """
    Implements Paper 2 EMR Model:
    4 Categorical features (Label Encoded) + 3 Textual fields (TF-IDF)
    concatenated, standardized, and classified via Linear SVM.
    """
    def __init__(self):
        self.cat_cols = ["gender", "age_bin", "race", "ethnicity"]
        self.text_cols = ["reason_for_admission", "diagnosis", "admit_order"]
        self.encoders = {col: LabelEncoder() for col in self.cat_cols}
        self.tfidf = TfidfVectorizer(max_features=50)
        self.scaler = StandardScaler()
        self.svm = SVC(kernel="linear", probability=True, random_state=42)

    def _transform_features(self, df_emr, fit=False):
        cat_matrix = np.zeros((len(df_emr), len(self.cat_cols)))
        for idx, col in enumerate(self.cat_cols):
            if fit:
                cat_matrix[:, idx] = self.encoders[col].fit_transform(df_emr[col].astype(str))
            else:
                cat_matrix[:, idx] = self.encoders[col].transform(df_emr[col].astype(str))

        combined_text = (
            df_emr["reason_for_admission"] + " " +
            df_emr["diagnosis"] + " " +
            df_emr["admit_order"]
        )
        if fit:
            text_matrix = self.tfidf.fit_transform(combined_text).toarray()
            concat = np.hstack([cat_matrix, text_matrix])
            return self.scaler.fit_transform(concat)
        else:
            text_matrix = self.tfidf.transform(combined_text).toarray()
            concat = np.hstack([cat_matrix, text_matrix])
            return self.scaler.transform(concat)

    def fit(self, df_emr, y):
        X_emr = self._transform_features(df_emr, fit=True)
        self.svm.fit(X_emr, y)
        return self

    def predict_proba(self, df_emr):
        X_emr = self._transform_features(df_emr, fit=False)
        return self.svm.predict_proba(X_emr)[:, 1]


# =====================================================================
# 5. STREAM 4: RESERVOIR-COMPUTING NEURAL NETWORK (RC-NN) FOR ECG (Paper 2)
# =====================================================================
class ReservoirReadoutANN(nn.Module):
    """Paper 2 3-layer Readout ANN with [20, 6] hidden neurons."""
    def __init__(self, n_reservoir=40):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_reservoir, 20),
            nn.Tanh(),
            nn.Linear(20, 6),
            nn.Tanh(),
            nn.Linear(6, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class ReservoirComputingECG:
    """
    Implements Paper 2 RC-NN Eq. (1):
    R_k[n] = tanh(G_i * W * X[n] + G_f * W_r * R_k[n-1])
    with optimal parameters: N=40, G_i=0.7, G_f=0.1, and W_r = Identity Matrix.
    """
    def __init__(self, n_reservoir=40, g_i=0.7, g_f=0.1):
        self.N = n_reservoir
        self.g_i = g_i
        self.g_f = g_f
        # Fixed untrainable random input weights W (N x 1) and Identity W_r (N x N)
        self.W_in = np.random.uniform(-1.0, 1.0, size=(self.N, 1))
        self.W_r = np.eye(self.N)
        self.scaler = StandardScaler()
        self.readout_ann = ReservoirReadoutANN(n_reservoir=self.N)

    def _project_reservoir(self, X_ecg):
        n_samples, D_steps = X_ecg.shape
        reservoir_states = np.zeros((n_samples, self.N))
        for i in range(n_samples):
            r_prev = np.zeros(self.N)
            for t in range(D_steps):
                x_t = X_ecg[i, t]
                r_curr = np.tanh(
                    self.g_i * (self.W_in[:, 0] * x_t) +
                    self.g_f * (self.W_r @ r_prev)
                )
                r_prev = r_curr
            reservoir_states[i] = r_prev
        return reservoir_states

    def fit(self, X_ecg, y, epochs=45, lr=0.005):
        R_states = self.scaler.fit_transform(self._project_reservoir(X_ecg))
        R_tensor = torch.tensor(R_states, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.readout_ann.parameters(), lr=lr)
        self.readout_ann.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            preds = self.readout_ann(R_tensor)
            loss = criterion(preds, y_tensor)
            loss.backward()
            optimizer.step()
        return self

    def predict_proba(self, X_ecg):
        self.readout_ann.eval()
        R_states = self.scaler.transform(self._project_reservoir(X_ecg))
        R_tensor = torch.tensor(R_states, dtype=torch.float32)
        with torch.no_grad():
            return self.readout_ann(R_tensor).squeeze(1).numpy()


# =====================================================================
# 6. LATE-FUSION META-CLASSIFIER (2-Layer ANN, 7 Hidden Neurons - Paper 2)
# =====================================================================
class LateFusionMetaLearner(nn.Module):
    """Combines prediction scores from all 4 specialist models using 7 hidden neurons."""
    def __init__(self, n_models=4, hidden_neurons=7):
        super().__init__()
        self.meta_net = nn.Sequential(
            nn.Linear(n_models, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, 1),
            nn.Sigmoid()
        )

    def forward(self, scores):
        return self.meta_net(scores)


# =====================================================================
# 7. EXECUTION & EVALUATION ENGINE
# =====================================================================
def evaluate_predictions(y_true, y_prob, name):
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "AUROC": roc_auc_score(y_true, y_prob),
        "AUPR": average_precision_score(y_true, y_prob)
    }


if __name__ == "__main__":
    print("Generating Multi-Modal Cardiovascular Cohort...")
    df_life, df_labs, df_emr, X_ecg, y = generate_multimodal_cohort(n_samples=1200)

    # Train/Test Split (80% Train, 20% Holdout Test - matching Paper 2's 1222/305 ratio)
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.20, stratify=y, random_state=42)

    # 1. Train & Evaluate Stream 1: Lifestyle XGB-LCE
    stream1 = LocalCascadeEnsembleClassifier(n_Select=15).fit(df_life.iloc[train_idx], y[train_idx])
    p_train_1 = stream1.predict_proba(df_life.iloc[train_idx])
    p_test_1 = stream1.predict_proba(df_life.iloc[test_idx])

    # 2. Train & Evaluate Stream 2: Lab 1D-CNN + Self-Attention
    stream2 = BiomarkerDeepPipeline(n_rf_keep=30, n_final_keep=26).fit(df_labs.iloc[train_idx], y[train_idx])
    p_train_2 = stream2.predict_proba(df_labs.iloc[train_idx])
    p_test_2 = stream2.predict_proba(df_labs.iloc[test_idx])

    # 3. Train & Evaluate Stream 3: EMR TF-IDF + Linear SVM
    stream3 = EMRPipeline().fit(df_emr.iloc[train_idx], y[train_idx])
    p_train_3 = stream3.predict_proba(df_emr.iloc[train_idx])
    p_test_3 = stream3.predict_proba(df_emr.iloc[test_idx])

    # 4. Train & Evaluate Stream 4: ECG Reservoir Computing (RC-NN)
    stream4 = ReservoirComputingECG(n_reservoir=40, g_i=0.7, g_f=0.1).fit(X_ecg[train_idx], y[train_idx])
    p_train_4 = stream4.predict_proba(X_ecg[train_idx])
    p_test_4 = stream4.predict_proba(X_ecg[test_idx])

    # 5. Train Late-Fusion Meta-Classifier (7 Hidden Neurons)
    S_train = torch.tensor(np.column_stack([p_train_1, p_train_2, p_train_3, p_train_4]), dtype=torch.float32)
    S_test = torch.tensor(np.column_stack([p_test_1, p_test_2, p_test_3, p_test_4]), dtype=torch.float32)
    y_train_t = torch.tensor(y[train_idx], dtype=torch.float32).unsqueeze(1)

    meta_model = LateFusionMetaLearner(n_models=4, hidden_neurons=7)
    meta_opt = optim.Adam(meta_model.parameters(), lr=0.01)
    meta_crit = nn.BCELoss()

    meta_model.train()
    for _ in range(120):
        meta_opt.zero_grad()
        out = meta_model(S_train)
        loss = meta_crit(out, y_train_t)
        loss.backward()
        meta_opt.step()

    meta_model.eval()
    with torch.no_grad():
        p_test_fusion = meta_model(S_test).squeeze(1).numpy()

    # Compile Results Table
    results = [
        evaluate_predictions(y[test_idx], p_test_1, "Stream 1: Lifestyle XGB-LCE (Paper 3)"),
        evaluate_predictions(y[test_idx], p_test_2, "Stream 2: Lab 1D-CNN + Self-Attention (Paper 1)"),
        evaluate_predictions(y[test_idx], p_test_3, "Stream 3: EMR TF-IDF + Linear SVM (Paper 2)"),
        evaluate_predictions(y[test_idx], p_test_4, "Stream 4: ECG Reservoir RC-NN (Paper 2)"),
        evaluate_predictions(y[test_idx], p_test_fusion, "CardioFuse-4 (Proposed Late-Fusion System)")
    ]
    results_df = pd.DataFrame(results)
    print("\n=== EXPERIMENTAL EXECUTION RESULTS ON HOLDOUT TEST SET ===")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
