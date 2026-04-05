from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import numpy as np
import os
import json
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

# ─── Global State ───────────────────────────────────────────
model_state = {
    'model': None,
    'label_encoder': None,
    'feature_cols': None,
    'label_map': None,
    'trained': False,
    'metrics': None,
    'importance': None,
    'ip_col': None,
    'flow_col': None,
}


# ─── ROUTES ─────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/train', methods=['POST'])
def train():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
    f.save(filepath)

    try:
        result = run_pipeline(filepath)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    if not model_state['trained']:
        return jsonify({'error': 'Model not trained yet'}), 400

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    f = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'predict_' + f.filename)
    f.save(filepath)

    try:
        result = run_prediction(filepath)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/model_info')
def model_info():
    if not model_state['trained']:
        return jsonify({'trained': False})
    return jsonify({
        'trained': True,
        'metrics': model_state['metrics'],
        'importance': model_state['importance'],
        'label_map': model_state['label_map'],
    })


# ─── PIPELINE ───────────────────────────────────────────────
def run_pipeline(filepath):
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    from lightgbm import LGBMClassifier

    df = pd.read_csv(filepath, low_memory=False)
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.duplicated()]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Detect columns
    if 'Tot Fwd Pkts' in df.columns:
        fwd_col, bwd_col = 'Tot Fwd Pkts', 'Tot Bwd Pkts'
    elif 'Total Fwd Packets' in df.columns:
        fwd_col, bwd_col = 'Total Fwd Packets', 'Total Backward Packets'
    else:
        raise ValueError("Packet columns not found")

    df['TotalPackets'] = df[fwd_col] + df[bwd_col]

    if 'Src IP' in df.columns:
        ip_col = 'Src IP'
    elif 'Source IP' in df.columns:
        ip_col = 'Source IP'
    else:
        raise ValueError("IP column not found")

    if 'Flow Byts/s' in df.columns:
        flow_col = 'Flow Byts/s'
    elif 'Flow Bytes/s' in df.columns:
        flow_col = 'Flow Bytes/s'
    else:
        flow_col = None

    model_state['ip_col'] = ip_col
    model_state['flow_col'] = flow_col

    # Fill NaN
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[num_cols] = df[num_cols].apply(lambda col: col.fillna(col.median()))
    df = df.dropna(subset=['Label'])

    # Aggregation
    conn_per_ip = df.groupby(ip_col).size().rename('TotalConnPerIP')
    avg_pkt = df.groupby(ip_col)['TotalPackets'].mean().rename('AvgPktPerFlowIP')
    df = df.merge(conn_per_ip, on=ip_col, how='left')
    df = df.merge(avg_pkt, on=ip_col, how='left')

    # Encode label
    le = LabelEncoder()
    df['Label'] = le.fit_transform(df['Label'])
    label_map = {int(v): k for k, v in zip(le.classes_, le.transform(le.classes_))}

    # Class distribution
    class_dist = df['Label'].value_counts().to_dict()
    class_dist_named = {label_map[k]: int(v) for k, v in class_dist.items()}

    # Drop non-numeric
    ip_series = df[ip_col].copy()
    drop_cols = ['Flow ID', 'Src IP', 'Dst IP', 'Source IP', 'Destination IP', 'Timestamp']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    for col in df.columns:
        if df[col].dtype == 'object':
            df = df.drop(columns=[col])

    X = df.drop(columns=['Label'])
    y = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LGBMClassifier(
        objective='multiclass',
        num_class=len(np.unique(y)),
        n_estimators=100,
        learning_rate=0.1,
        verbose=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, target_names=[label_map[i] for i in sorted(label_map)], output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(15)

    # Store state
    model_state['model'] = model
    model_state['label_encoder'] = le
    model_state['feature_cols'] = list(X.columns)
    model_state['label_map'] = label_map
    model_state['trained'] = True

    metrics = {
        'accuracy': float(report['accuracy']),
        'report': {k: v for k, v in report.items() if isinstance(v, dict)},
        'confusion_matrix': cm,
        'cm_labels': [label_map[i] for i in sorted(label_map)],
        'class_distribution': class_dist_named,
        'n_samples': int(len(df)),
        'n_features': int(len(X.columns)),
    }
    model_state['metrics'] = metrics
    model_state['importance'] = importance.to_dict('records')

    return {
        'success': True,
        'metrics': metrics,
        'importance': importance.to_dict('records'),
        'label_map': label_map,
    }


def run_prediction(filepath):
    model = model_state['model']
    le = model_state['label_encoder']
    feature_cols = model_state['feature_cols']
    label_map = model_state['label_map']
    ip_col = model_state['ip_col']
    flow_col = model_state['flow_col']

    df = pd.read_csv(filepath, low_memory=False)
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.duplicated()]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    if 'Tot Fwd Pkts' in df.columns:
        fwd_col, bwd_col = 'Tot Fwd Pkts', 'Tot Bwd Pkts'
    elif 'Total Fwd Packets' in df.columns:
        fwd_col, bwd_col = 'Total Fwd Packets', 'Total Backward Packets'
    else:
        fwd_col = bwd_col = None

    if fwd_col:
        df['TotalPackets'] = df[fwd_col] + df[bwd_col]

    if ip_col in df.columns:
        conn_per_ip = df.groupby(ip_col).size().rename('TotalConnPerIP')
        avg_pkt = df.groupby(ip_col)['TotalPackets'].mean().rename('AvgPktPerFlowIP') if 'TotalPackets' in df.columns else None
        df = df.merge(conn_per_ip, on=ip_col, how='left')
        if avg_pkt is not None:
            df = df.merge(avg_pkt, on=ip_col, how='left')

    ip_series = df[ip_col].copy() if ip_col in df.columns else None

    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[num_cols] = df[num_cols].apply(lambda col: col.fillna(col.median()))

    # Align features
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    X = df[feature_cols]

    preds = model.predict(X)
    pred_labels = [label_map.get(int(p), str(p)) for p in preds]

    result_df = pd.DataFrame({'Predicted': pred_labels})
    if ip_series is not None:
        result_df[ip_col] = ip_series.values[:len(result_df)]

    if flow_col and flow_col in df.columns:
        result_df[flow_col] = df[flow_col].values[:len(result_df)]

    summary = result_df['Predicted'].value_counts().to_dict()
    attacks = result_df[result_df['Predicted'] != 'BENIGN']

    top_attackers = []
    if ip_col in result_df.columns:
        top_attackers = (
            attacks.groupby(ip_col)['Predicted'].count()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
            .rename(columns={'Predicted': 'Count', ip_col: 'IP'})
            .to_dict('records')
        )

    sample_attacks = attacks.head(50).fillna('').to_dict('records')

    return {
        'success': True,
        'summary': summary,
        'top_attackers': top_attackers,
        'sample_attacks': sample_attacks,
        'total': len(result_df),
        'attack_count': int(len(attacks)),
        'benign_count': int(len(result_df) - len(attacks)),
    }


if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True, port=5000)
