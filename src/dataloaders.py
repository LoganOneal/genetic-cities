import pandas as pd

def load_zone_info(file_path):
    zones = pd.read_csv(file_path)
    types = zones["type"].to_numpy()
    min_areas = zones["min_area"].to_numpy()
    min_widths = zones["min_width"].to_numpy()

    return types, min_areas, min_widths

def load_building_info(file_path):
    buildings = pd.read_csv(file_path)
    types = buildings["name"].to_numpy()
    min_areas = buildings["min_area"].to_numpy()
    min_widths = buildings["min_width"].to_numpy()
    required_zone = buildings["required_zone"].to_numpy()

    return types, min_areas, min_widths, required_zone