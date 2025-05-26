import os
import numpy as np
from setup_utils import parse_arguments

# STAGE          = "stage_severe_pred"
# NEPOCH         = "990"
# MODEL_PREFIX   = "mod_st"
# SEEDS          = [1001, 138647, 193, 34, 44] #['1001','138647','193','34','44','71530','875688','8765','985772','244959']

STAGE          = "ss"
NEPOCH         = "400"
MODEL_PREFIX   = "mod_st"
SEEDS          = ['1001','138647','193','34','44','71530','875688','8765','985772','244959']

def average_seeded_features(
    feature_root: str,
    stage: str,
    model_prefix: str,
    nepoch: str,
    seeds: list[str],
    output_root: str
):
    """
    For each split ('train','test'), each class label folder, and
    each filename.npy, load that file from every seed subfolder,
    average the feature vectors, and save to output_root in the
    same subfolder structure.
    """
    base_dir = os.path.join(feature_root, stage)
    # find all the seed‐specific folders for this model+epoch
    seed_dirs = [
        d for d in os.listdir(base_dir)
        if (model_prefix in d) 
        and (f'epoch_{nepoch}' in d)
        and any(f'seed_{s}' in d for s in seeds)
    ]
    if not seed_dirs:
        raise RuntimeError(f"No seed dirs found in {base_dir} matching prefix/epoch")

    for split in ("train", "test"):
        for label in sorted(os.listdir(os.path.join(base_dir, seed_dirs[0], split))):
            inp_label_dirs = [
                os.path.join(base_dir, sd, split, label)
                for sd in seed_dirs
            ]
            out_label_dir = os.path.join(output_root,  split, label)
            os.makedirs(out_label_dir, exist_ok=True)

            # assume all seed dirs have the same filenames in each label
            filenames = os.listdir(inp_label_dirs[0])

            for fname in filenames:
                # load every seed's feature vector
                feats = []
                for d in inp_label_dirs:
                    path = os.path.join(d, fname)
                    feats.append(np.load(path))
                # stack into shape (n_seeds, feature_dim) and average
                mean_feat = np.mean(np.stack(feats, axis=0), axis=0)
                # save
                out_path = os.path.join(out_label_dir, fname)
                np.save(out_path, mean_feat)


if __name__ == "__main__":
    try:
       args = parse_arguments()
       print(f"Arguments: {args}")
    except Exception as e:
       print("CRITICAL ERROR DURING ARGUMENT PARSING.")
       print(e)
       raise

    MODEL_PREFIX= args.model_name
    FEATURE_ROOT   = args.feature_save_path   # same as args.feature_save_path
    OUTPUT_ROOT    = os.path.join(FEATURE_ROOT, STAGE, f'average_{MODEL_PREFIX}')  # wherever you want to dump the means
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    average_seeded_features(
        feature_root=FEATURE_ROOT,
        stage=STAGE,
        model_prefix=MODEL_PREFIX,
        nepoch=NEPOCH,
        seeds=SEEDS,
        output_root=OUTPUT_ROOT
    )
    print("Done averaging features across seeds.")
