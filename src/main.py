import pandas as pd
    
from dataloaders import load_zone_info

W, L = 100, 100

type, min_areas, min_widths = load_zone_info("./data/zones.csv")


def main():
    print("Hello, World!")



if __name__ == "__main__":
    main()