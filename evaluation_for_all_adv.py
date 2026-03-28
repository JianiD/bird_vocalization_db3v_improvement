import os
import subprocess
import json
import pandas as pd

models = [
        "tdnn_IFN_aug_sampler_adv",
        "tdnn_IFN_2_CycleGAN_aug_sampler_adv",
        "tdnn_IFN_3_aug_sampler_adv",
]
datasets = ["meta-v02/1", "meta-v02/2", "meta-v02/3"]


template_cfg = "Config/config_tdnn.yaml"
temp_cfg     = "Config/config_tdnn_temp.yaml"

summary_file = "summary.csv"
rows = []

for model in models:
    for ds in datasets:
        print(f"\n=== Evaluating model {model} on {ds} ===")

        if os.path.exists("eval_result_adv.json"):
            os.remove("eval_result_adv.json")

        with open(template_cfg) as f:
            cfg_txt = f.read()

        tdnn_type = model.split("_")[0] + "_" + model.split("_")[1]
        result_path = f"results/{model}/local/ckpt"
        cfg_txt = (cfg_txt
                .replace("<TDNN_TYPE>", tdnn_type)
                .replace("<EVAL_DS>",   ds)
                .replace("<RESULT_DIR>", result_path))


        with open(temp_cfg, "w") as f:
            f.write(cfg_txt)

        try:
            subprocess.run([
                "python", "evaluation_tdnn_adv.py",
                f"model.tdnn={tdnn_type}",
                f"meta.result={result_path}",
                f"evaluation.ds={ds}",
                "hydra.run.dir=.",
                "hydra.job.chdir=False",
                f"hydra.job.name=eval_{model}_{ds.replace('/', '_')}"
            ], check=True)


        except subprocess.CalledProcessError as e:
            print(f"Failed on {model} - {ds}")
            print(e.output)    
            print(f"Return code: {e.returncode}")
            print(f"Command: {e.cmd}")
            rows.append((model, ds, "FAIL", "FAIL", "FAIL"))
            continue

        
        if not os.path.exists("eval_result_adv.json"):
            print("Result file not found")
            rows.append((model, ds, "N/A", "N/A", "N/A"))
            continue

        try:
            with open("eval_result_adv.json") as f:
                content = f.read()
            print("DEBUG: JSON content ->", content) 
            data = json.loads(content)
            rows.append((model, ds,
                         data.get("acc", "N/A"),
                         data.get("uar", "N/A"),
                         data.get("f1",  "N/A")))
        except json.JSONDecodeError:
            print("Failed to decode JSON result")
            rows.append((model, ds, "N/A", "N/A", "N/A"))


pd.DataFrame(rows, columns=["Model", "Dataset", "Acc", "UAR", "F1"]) \
  .to_csv(summary_file, index=False)

print(f"\n Summary written to {summary_file}")
