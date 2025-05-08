import argparse

def main():
    parser = argparse.ArgumentParser(description="Train or Test LSTMModel")
    parser.add_argument("--mode", choices=["train", "test"], required=True, help="Mode: train or test")
    args = parser.parse_args()

    if args.mode == "train":
        from train.train_main import train_main
        train_main()
    elif args.mode == "test":
        from test.test_main import test_main
        test_main()

if __name__ == "__main__":
    main()
