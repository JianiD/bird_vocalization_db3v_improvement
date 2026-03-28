import os
import subprocess
import json
import pandas as pd

# 模型和数据集配置
models = [
    # "tdnn_IFN_2_Aug_Mixup_CycleGAN_w0.1_new",
    #       "tdnn_IFN_2_Aug_Mixup_CycleGAN_w0.3_new",
    #       "tdnn_IFN_2_Aug_Mixup_CycleGAN_w0.5_new",
          "tdnn_IFN_2_Aug_Mixup_CycleGAN_w1"]
datasets = ["meta-v02/1", "meta-v02/2_CycleGAN", "meta-v02/3"]

template_cfg = "Config/config_tdnn.yaml"
temp_cfg     = "Config/config_tdnn_temp.yaml"

summary_file = "Summary_with_soundscape_noise_1_weight.csv"
rows = []

# Noise
noise_conditions = ["White_5dB", "White_10dB", "White_25dB", "White_35dB", "Field_10dB"]

for model in models:
    for ds in datasets:
        print(f"\n=== Evaluating model {model} on {ds} ===")

        if os.path.exists("eval_result_adv.json"):
            os.remove("eval_result_adv.json")

        with open(template_cfg) as f:
            cfg_txt = f.read()

        parts = model.split("_")
        tdnn_type = parts[0] + "_" + parts[1]
        result_path = f"results/{model}/local/ckpt"
        
        cfg_txt = (cfg_txt
                .replace("<TDNN_TYPE>", tdnn_type)
                .replace("<EVAL_DS>",   ds)
                .replace("<RESULT_DIR>", result_path))

        with open(temp_cfg, "w") as f:
            f.write(cfg_txt)

        try:
            subprocess.run([
                "python", "evaluation_tdnn_adv_noise.py",
                f"model.tdnn={tdnn_type}",
                f"meta.result={result_path}",
                f"evaluation.ds={ds}",
                "hydra.run.dir=.",
                "hydra.job.chdir=False"
            ], check=True)

        except subprocess.CalledProcessError as e:
            print(f"Failed on {model} - {ds}")
            continue
        
        if not os.path.exists("eval_result_adv.json"):
            print("Result file not found")
            continue

        try:
            with open("eval_result_adv.json") as f:
                data = json.load(f)
            
            row = {
                "Model": model,
                "Dataset": ds,
                "Clean_Acc": data.get("acc", "N/A"),
                "Clean_UAR": data.get("uar", "N/A"),
                "Clean_F1":  data.get("f1",  "N/A")
            }
            

            robustness = data.get("robustness", {})
            for cond in noise_conditions:
                val = robustness.get(cond, {}).get("acc", "N/A")
                row[f"{cond}_Acc"] = val
                
            rows.append(row)
            
        except Exception as e:
            print(f"Error parsing results: {e}")


df = pd.DataFrame(rows)
cols = ["Model", "Dataset", "Clean_Acc", "Clean_UAR", "Clean_F1"] + [f"{c}_Acc" for c in noise_conditions]
df = df[cols] 

df.to_csv(summary_file, index=False)
print(f"\nComprehensive summary written to {summary_file}")