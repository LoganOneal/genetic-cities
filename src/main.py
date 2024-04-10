import pandas as pd
    
from dataloaders import load_zone_info, load_building_info

W, L = 1000, 1000

zone_types, zone_min_areas, zone_min_widths = load_zone_info("./data/zones.csv")
zone_min_areas = 0.25 * zone_min_areas
max_area_weight = 1

building_types, building_min_areas, building_min_widths, required_zone = load_building_info("./data/buildings.csv")


def main():
    print("Hello, World!")



if __name__ == "__main__":
    main()