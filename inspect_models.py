import joblib
import json
import warnings
warnings.filterwarnings('ignore')

models = {
    "adaboost": r"c:\Users\Vaka Bela Jithendra\OneDrive\Desktop\ML Algorithm Explorer\Supervised Learning\AdaBoostRegression\insurance_charges_full_pipeline.pkl",
    "catboost": r"c:\Users\Vaka Bela Jithendra\OneDrive\Desktop\ML Algorithm Explorer\Supervised Learning\CatGBMRegression\house_price_full_pipeline.pkl",
    "xgboost": r"c:\Users\Vaka Bela Jithendra\OneDrive\Desktop\ML Algorithm Explorer\Supervised Learning\XGboostRegression\podcast_listening_full_pipeline.pkl"
}

output = {}

for name, model_path in models.items():
    res = {"target": "", "num": [], "cat": []}
    try:
        pipeline = joblib.load(model_path)
        if isinstance(pipeline, dict):
            if 'scaler' in pipeline:
                scaler = pipeline['scaler']
                if hasattr(scaler, 'feature_names_in_'):
                    res["num"] = list(scaler.feature_names_in_)
            if 'label_encoders' in pipeline:
                le = pipeline['label_encoders']
                if isinstance(le, dict):
                    res["cat"] = list(le.keys())
    except Exception as e:
        res["error"] = str(e)
    output[name] = res

# write to file
with open('features.json', 'w') as f:
    json.dump(output, f, indent=4)
