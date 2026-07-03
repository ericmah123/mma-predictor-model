"""Full pipeline: raw CSVs -> training data + fighter DB -> trained model."""

from mma_predictor.models import train
from mma_predictor.models.preprocess import DATA_DIR, build_dataset, save_fighter_db


def main():
    df, snapshots = build_dataset()
    df.to_csv(f"{DATA_DIR}/training_data.csv", index=False)
    save_fighter_db(snapshots)
    print(f"Built {len(df)} training rows and {len(snapshots)} fighter snapshots.")
    train.main()


if __name__ == "__main__":
    main()
