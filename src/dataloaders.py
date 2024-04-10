import pandas as pd

def load_zone_info(file_path):
    zones = pd.read_csv(file_path)
    type = zones["type"].to_numpy()
    min_areas = zones["min_area"].to_numpy()
    min_widths = zones["min_width"].to_numpy()

    return type, min_areas, min_widths