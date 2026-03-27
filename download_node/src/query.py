
from astroquery.gaia import Gaia

class GaiaQueryWrapper:
   
    def __init__(self):
        pass

    def send_gaia_query(parallax_lower_bound:int = 0, parallax_over_error_lower_bound: int = 5, ruwe_upper_bound: float = 1.4, phot_g_mean_mag_upper_bound: int = 8):
        query = f"""
            SELECT
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
                AND phot_g_mean_mag < {phot_g_mean_mag_upper_bound}
        """

        job = Gaia.launch_job(query)
        return job.get_results()
    

if __name__ == "__main__":
    # Test patch — solar neighborhood, runs synchronously
    print("Running test query...")
    gqh = GaiaQueryWrapper()
    results = gqh.send_gaia_query(10, 5, 1.4, 21) # Test Query
    # results = send_gaia_query() # Full Query

    print(f"Query returned {len(results)} stars")
    print(results[:5])

    write_results_to_file = True

    if write_results_to_file:
      results.write("results.csv", format="pandas.csv")
