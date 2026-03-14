import json
import re
import os

BASE = r"c:\Users\Vaka Bela Jithendra\OneDrive\Desktop\ML Algorithm Explorer\Supervised Learning"

notebooks = {
    "bayesian":     os.path.join(BASE, "BayesianRegression", "energy-efficiency-bayesian-regression.ipynb"),
    "elasticnet":   os.path.join(BASE, "ElasticNetRegression", "bike-sharing-elasticnet-regression.ipynb"),
    "lasso":        os.path.join(BASE, "LassoRegression", "video-game-sales-lasso-regression.ipynb"),
    "linear":       os.path.join(BASE, "LinearRegression", "House Price Prediction - Linear Regression.ipynb"),
    "polynomial":   os.path.join(BASE, "PolynomialRegression", "manufacturing-polynomial-regression.ipynb"),
    "randomforest": os.path.join(BASE, "RandomForestRegression", "car-price-prediction-randomforestregressor.ipynb"),
    "ridge":        os.path.join(BASE, "RidgeRegression", "CryptoCurrency Prediction - Ridge Regression.ipynb"),
    "svr":          os.path.join(BASE, "SupportVectorRegression", "Stock Price- Support Vector Regressor.ipynb"),
    "adaboost":     os.path.join(BASE, "AdaBoostRegression", "insurance-charges1-adaboost-regressor.ipynb"),
    "catboost":     os.path.join(BASE, "CatGBMRegression", "advanced-house-price-catgbm.ipynb"),
    "xgboost":      os.path.join(BASE, "XGboostRegression", "podcast-listening-xgboost-regressor.ipynb"),
}

results = {}

for name, nb_path in notebooks.items():
    info = {"notebook": nb_path, "dataset": "", "target": "", "features_dropped": [], "imports": []}
    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
            code = ""
            for cell in nb.get("cells", []):
                if cell["cell_type"] == "code":
                    code += "".join(cell.get("source", [])) + "\n"

            # find dataset loading
            for line in code.split("\n"):
                if "read_csv" in line or "load_" in line:
                    info["dataset"] = line.strip()[:120]
                    break

            # find target
            for line in code.split("\n"):
                if "y =" in line or "y=" in line:
                    info["target"] = line.strip()[:120]
                    break

            # find drop columns
            for line in code.split("\n"):
                if "drop" in line and ("columns" in line or "axis" in line):
                    info["features_dropped"].append(line.strip()[:120])

            # find .columns output (first one)
            for line in code.split("\n"):
                if ".columns" in line and "print" not in line and "#" not in line:
                    info["imports"].append(line.strip()[:120])
                    if len(info["imports"]) >= 3:
                        break

    except Exception as e:
        info["error"] = str(e)

    results[name] = info

with open("all_notebooks_info.json", "w") as f:
    json.dump(results, f, indent=2)

print("Done! Wrote all_notebooks_info.json")
