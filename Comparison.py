import pandas as pd
import os

def calculate_tracking_metrics(video_label, baseline_path, deepsort_path):
    # folder final_results
    if not os.path.exists(baseline_path) or not os.path.exists(deepsort_path):
        return None

    # load the files
    df_baseline = pd.read_csv(baseline_path, header=None)
    df_deepsort = pd.read_csv(deepsort_path, header=None)

    # calculate the unique id's
    ids_baseline = df_baseline[1].nunique()
    ids_deepsort = df_deepsort[1].nunique()

    # extraction of the % of the reduction
    reduction = ((ids_baseline - ids_deepsort) / ids_baseline) * 100

    return {
        "Video": video_label,
        "Baseline IDs": ids_baseline,
        "DeepSORT IDs": ids_deepsort,
        "Improvement": round(reduction, 2)
    }

# comparison list
comparisons = [
    ("Urban Traffic (videoplayback)", "sort_results_x_traffic.txt", "my_results_x_traffic.txt"),
    ("Football Match (08fd33_4)", "sort_results_x_football.txt", "my_results_x_football.txt")
]

# print the resultss
print("\n" + "-" * 85)
print(f"{'Video Content':<30} | {'Baseline (IDs)':<15} | {'DeepSORT (IDs)':<15} | {'Reduction'}")
print("-" * 85)

for label, base, deep in comparisons:
    res = calculate_tracking_metrics(label, base, deep)
    if res:
        print(f"{res['Video']:<30} | {res['Baseline IDs']:<15} | {res['DeepSORT IDs']:<15} | {res['Improvement']}%")
    else:
        print(f"⚠️ Τα αρχεία για το '{label}' δεν βρέθηκαν.")

print("-" * 85 + "\n")
print("✅ Η ανάλυση ολοκληρώθηκε!")