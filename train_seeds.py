from train import main

if __name__ == "__main__":
    seeds = [42, 1337, 564738]
    for s in seeds:
        print("Start for seed", s)
        try:
            main(s)
        except Exception as e:
            print(f"!! Skipping {s} - {e}")
