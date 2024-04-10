import pandas as pd
    
from dataloaders import load_zone_info, load_building_info


'''
"The nearness preferences are on a six-point scale, from
 A=Absolutely Necessary and X = Not Desirable. The cells are one-indexed." 
 (https://profiles.stanford.edu/griffin-holt?tab=research-and-scholarship)
'''
NEARNESS_SCALE = {"A": 1, "E": 0, "I": 0, "O": 0, "U": 0, "X": 0}

W, L = 1000, 1000

zone_types, zone_min_areas, zone_min_widths = load_zone_info("./data/zones.csv")
zone_min_areas = 0.25 * zone_min_areas
zone_max_area_weight = 1

building_types, building_min_areas, building_min_widths, required_zone = load_building_info("./data/buildings.csv")
building_man_area = 0.25 * building_min_areas
building_max_area = 1.5 * building_min_areas

relationship_graph, must_be_close = load_relationship_graph(N, "./data/relationship_chart.csv")

def main():
    print("Hello, World!")



if __name__ == "__main__":
    main()