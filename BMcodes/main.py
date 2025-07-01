from pathlib import Path
from pose_compare import extract_all_angles_and_frames, compare_with_dtw, generate_dtw_video

shot_type1 = 'front_left_lift'
shot_num1 = '01'
shot_type2 = 'back_left_lift'
shot_num2 = '01'

shot_det_1 = f"{shot_type1}_{shot_num1}"
shot_det_2 = f"{shot_type2}_{shot_num2}"

res = {
    "Shot Type1": shot_det_1,
    "Shot Type2": shot_det_2
}

script_dir = Path(__file__).parent
data_dir = script_dir / "data"
f1 = data_dir / f"{shot_det_1}.mp4"
f2 = data_dir / f"{shot_det_2}.mp4"
out_filename = script_dir / "outputs" / "comparison_output.mp4"

vecs1, frames1 = extract_all_angles_and_frames(f1)
vecs2, frames2 = extract_all_angles_and_frames(f2)
path = compare_with_dtw(vecs1, vecs2)
res.update(generate_dtw_video(vecs1, frames1, vecs2, frames2, path, out_filename))

print("=" * 60)
print("\nFINAL VERDICT REPORT")
print("~" * 20)
print(f"\nShot 1: {res['Shot Type1']}")
print(f"Shot 1: {res['Shot Type2']}\n")
print(f"Average Euclidean Distance: {res['Average Euclidean']}")
print(f"Average Cosine Distance: {res['Average Cosine']}\n")
print(f"Average Euclidean Score: {res['Euclidean Score']}")
print(f"Average Cosine Score: {res['Cosine Score']}\n")
print(f"Total Score: {res['Total Score']}\n")
print(f"FINAL VERDICT: {res['Verdict']}")