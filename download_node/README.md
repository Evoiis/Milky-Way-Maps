# MWM Download Node


# Testing

run `pytest` in this folder


## Data Labels and Descriptions


| Column | Description | Range |
|---|---|---|
| `source_id` | Gaia's unique ID number for this star | integer |
| `ra` | How far around the sky horizontally (like longitude, but for space) | 0 – 360° |
| `dec` | How far up or down in the sky (like latitude, but for space) | -90 – 90° |
| `parallax` | How much the star appears to shift as Earth orbits the Sun — bigger = closer. Divide 1000 by this to get distance in parsecs | 0 – 1000 mas |
| `pmra` | How fast the star is drifting left/right across the sky per year | ~-10000 – 10000 mas/yr |
| `pmdec` | How fast the star is drifting up/down across the sky per year | ~-10000 – 10000 mas/yr |
| `pmra_error` | How uncertain the left/right drift measurement is | 0+ |
| `pmdec_error` | How uncertain the up/down drift measurement is | 0+ |
| `phot_g_mean_mag` | How bright the star looks from Earth. Lower = brighter | ~3 – 21 |
| `bp_rp` | Star color. Low = blue-white hot star, high = red cool star | ~-0.5 – 5.0 |
| `ruwe` | Data quality score. Below 1.4 means clean measurement | 0+ |
| `radial_velocity` | How fast the star is moving toward or away from us in km/s. Negative = coming toward us. NULL for most stars | ~-1000 – 1000 km/s |
| `teff_gspphot` | Surface temperature in Kelvin. Sun is ~5778K, blue stars 20000K+, red dwarfs 3000K | ~3000 – 50000 K |
| `logg_gspphot` | Surface gravity. Low = bloated giant, high = compact dwarf | ~0 – 6 |
| `lum_flame` | True brightness regardless of distance, in multiples of the Sun's brightness | ~0.001 – 10^6 L☉ |
| `radius_flame` | Physical size compared to the Sun. 100 = 100x wider than the Sun | ~0.1 – 1000 R☉ |


