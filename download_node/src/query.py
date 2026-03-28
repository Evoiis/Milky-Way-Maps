import pandas as pd
import numpy as np
from astroquery.gaia import Gaia
from astropy.table import Table

class GaiaQueryWrapper:
   
    def __init__(self):
        pass

    def get_gaia_data(self):
        data = self._send_gaia_query()

        df = data.to_pandas()

        return df

    def _read_from_file(self):
        pass

    def _write_to_file(self, df: pd.DataFrame):
        pass
    


    def _send_gaia_query(
            self,
            parallax_lower_bound:int = 0.3,
            parallax_over_error_lower_bound: int = 5,
            ruwe_upper_bound: float = 1.4,
            phot_g_mean_mag_upper_bound: int = 15
        ) -> Table:
        """
        
        Output:
        Table with:
        source_id,ra,dec,parallax,pmra,pmdec,pmra_error,pmdec_error,phot_g_mean_mag,bp_rp,ruwe,radial_velocity,teff_gspphot,logg_gspphot,lum_flame,radius_flame
        """

        query = f"""
            SELECT TOP 500000
                g.source_id,
                g.ra, g.dec,
                g.parallax,
                g.pmra, g.pmdec,
                g.pmra_error, g.pmdec_error,
                g.phot_g_mean_mag,
                g.bp_rp,
                g.ruwe,
                g.radial_velocity,
                g.teff_gspphot,
                g.logg_gspphot,
                ap.lum_flame,
                ap.radius_flame
            FROM gaiadr3.gaia_source g
            LEFT JOIN gaiadr3.astrophysical_parameters ap
                ON g.source_id = ap.source_id
            WHERE
                g.parallax > {parallax_lower_bound}
                AND g.parallax_over_error > {parallax_over_error_lower_bound}
                AND g.ruwe < {ruwe_upper_bound}
                AND g.phot_g_mean_mag < {phot_g_mean_mag_upper_bound}
                AND g.bp_rp IS NOT NULL
                AND MOD(CAST(g.random_index AS BIGINT), 50) = 0
                AND g.radial_velocity IS NOT NULL
        """

        job = Gaia.launch_job(query)
        return job.get_results()
    


if __name__ == "__main__":
    # Test patch — solar neighborhood, runs synchronously
    print("Running test query...")
    gqw = GaiaQueryWrapper()
    results = gqw._send_gaia_query(100, 5, 1.4, 21) # Test Query
    df = results.to_pandas()
    df = gqw._calculate_cartesian_coordinates(df)

    df.to_csv("df.csv")

    print(f"Query returned {len(results)} stars")
    print(results[:5])

    write_results_to_file = False

    if write_results_to_file:
      results.write("results.csv", format="pandas.csv")
