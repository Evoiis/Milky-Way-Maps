import pandas as pd
import numpy as np

class GaiaDataProcessor():

    def __init__(self):
        pass

    def process_data(self, df: pd.DataFrame):
        pass

    def _calculate_cartesian_coordinates(self, df: pd.DataFrame):
        df["ra_rad"] = np.deg2rad(df["ra"].values)
        df["dec_rad"] = np.deg2rad(df["dec"].values)

        parsecs = 1000 / df["parallax"]

        df["pos_x"] = parsecs * np.cos(df["dec_rad"]) * np.cos(df["ra_rad"])
        df["pos_y"] = parsecs * np.cos(df["dec_rad"]) * np.sin(df["ra_rad"])
        df["pos_z"] = parsecs * np.sin(df["dec_rad"])

    def _calculate_rgb_color(self, df: pd.DataFrame):
        # bp_rp -> color
        # red -> white -> blue
        # -0.5 --> 2.5 -> 5.0

        bp_rp = np.clip(df["bp_rp"], -0.5, 5.0)
        
        white_point = 2.5 / 5.5
        norm = (bp_rp + 0.5)/ 5.5

        low_norm = norm / white_point
        high_norm = (norm - white_point)/(1 - white_point)
        
        df["color_r"] = np.where(
            norm < white_point,
            low_norm * 255,
            255
        )

        df["color_g"] = np.where(
            norm < white_point,
            100 + (low_norm * 155),
            255 - high_norm * 225
        )

        df["color_b"] = np.clip(
            np.where(
                norm <= white_point,
                255,
                255 - high_norm * 255 * 2
            ),
            0,
            255
        )

    def _calculate_star_brightness(self, df: pd.DataFrame):

        # teff_gspphot, surface temp
        # ~3000 RED --> 50000 BLUE
        # lum_flame
        # phot_g_mean_mag
        pass

    def _calculate_star_size(self, df:pd.DataFrame):

        # teff_gspphot, surface temp
        # ~3000 RED --> 50000 BLUE

        # fallback bprp
        pass
