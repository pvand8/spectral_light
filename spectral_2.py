import xarray as xr
import numpy as np

# --------------------------------------------------
# Input / Output files
# --------------------------------------------------
infile  = "era5_sw_2024_02_german_baltic_smallbox.nc"
outfile = "spectral_PAR_era5_2024_02.nc"

# --------------------------------------------------
# Constants
# --------------------------------------------------
dt = 3600.0      # seconds (ERA5 hourly accumulation)
f_par = 0.43     # PAR fraction of shortwave
dlambda = 5.0    # nm band width (metadata)

# --------------------------------------------------
# ROMS spectral wavelengths (402..697 nm, step 5 nm)
# --------------------------------------------------
wavelength_vals = np.arange(402.0, 702.0, 5.0)   # length 60

# --------------------------------------------------
# ROMS spectral shape (ec_Fobar) from mod_eclight.F (60 values)
# --------------------------------------------------
ec_Fobar = np.array([
    1.7990e-01, 1.6810e-01, 1.8120e-01, 1.7150e-01, 1.7600e-01,
    1.6730e-01, 1.7270e-01, 1.8450e-01, 1.8920e-01, 1.8740e-01,
    2.0040e-01, 2.0600e-01, 2.1010e-01, 1.9660e-01, 2.0220e-01,
    1.9960e-01, 2.0540e-01, 1.7230e-01, 1.9090e-01, 2.0030e-01,
    1.8900e-01, 1.9770e-01, 1.9130e-01, 1.6880e-01, 1.8910e-01,
    1.8340e-01, 1.9480e-01, 1.8610e-01, 1.8470e-01, 1.8670e-01,
    1.8700e-01, 1.8260e-01, 1.8600e-01, 1.8440e-01, 1.8840e-01,
    1.8570e-01, 1.8660e-01, 1.8080e-01, 1.7970e-01, 1.8100e-01,
    1.7290e-01, 1.7490e-01, 1.7440e-01, 1.7050e-01, 1.7040e-01,
    1.6890e-01, 1.6540e-01, 1.6620e-01, 1.6160e-01, 1.6110e-01,
    1.6000e-01, 1.3770e-01, 1.5810e-01, 1.5500e-01, 1.5190e-01,
    1.4880e-01, 1.4660e-01, 1.4500e-01, 1.4350e-01, 1.4200e-01
], dtype=float)

# --------------------------------------------------
# Sanity checks + normalize weights
# --------------------------------------------------
if ec_Fobar.size != wavelength_vals.size:
    raise ValueError(f"ec_Fobar has {ec_Fobar.size} values but wavelength has {wavelength_vals.size}")

weights = ec_Fobar / ec_Fobar.sum()

if not np.isfinite(weights).all():
    raise ValueError("Weights contain NaN/Inf")
if (weights < 0).any():
    raise ValueError("Weights contain negative values")

w_sum = float(weights.sum())
print("Weights OK:")
print("sum =", w_sum)
print("min =", float(weights.min()), "max =", float(weights.max()))

# --------------------------------------------------
# Open ERA5 file
# --------------------------------------------------
ds = xr.open_dataset(infile)

# Drop expver to avoid merge conflicts (ERA5 peculiarity)
ds = ds.drop_vars("expver", errors="ignore")

# Hagen (a): rename valid_time -> time
ds = ds.rename({"valid_time": "time"})

ssr = ds["ssr"]  # accumulated J m-2
print("Input ssr dims:", ssr.dims)
print("Shape:", ssr.shape)

# --------------------------------------------------
# Convert accumulated SSR -> flux SW (W m-2)
# --------------------------------------------------
# Incremental energy per step
dssr = ssr.diff("time")

# Pad first timestep back to original length
dssr = dssr.reindex(time=ssr.time)
dssr[0, :, :] = 0.0

# Remove small negative noise
dssr = dssr.clip(min=0.0)

SW = dssr / dt
print("SW min/max:", float(SW.min()), float(SW.max()))

# --------------------------------------------------
# Convert SW -> total PAR (W m-2)
# --------------------------------------------------
PAR_total = f_par * SW
print("PAR_total min/max:", float(PAR_total.min()), float(PAR_total.max()))

# --------------------------------------------------
# Create wavelength coordinate
# --------------------------------------------------
wavelength = xr.DataArray(
    wavelength_vals,
    dims=["wavelength"],
    name="wavelength",
    attrs={"units": "nm", "long_name": "wavelength"}
)

# weights as DataArray with wavelength coord
weights_da = xr.DataArray(
    weights,
    dims=["wavelength"],
    coords={"wavelength": wavelength},
    name="weights",
    attrs={"units": "1", "long_name": "Normalized spectral weights"}
)

# --------------------------------------------------
# Build spectral PAR per band
# --------------------------------------------------
par_band = PAR_total * weights_da

# Hagen (b): enforce axis order in NetCDF
par_band = par_band.transpose("time", "latitude", "longitude", "wavelength")

par_band.name = "par_band"
par_band.attrs.update({
    "units": "W m-2",
    "long_name": "Spectral PAR per wavelength band",
    "description": "Band-integrated spectral PAR. Sum over wavelength equals PAR_total."
})

# --------------------------------------------------
# Hagen (c): add band width variable (nm)
# --------------------------------------------------
par_band_width = xr.DataArray(
    np.full(len(wavelength_vals), dlambda, dtype=float),
    dims=["wavelength"],
    coords={"wavelength": wavelength},
    name="par_band_width",
    attrs={"units": "nm", "long_name": "Wavelength band width"}
)

# --------------------------------------------------
# Output dataset (store only par_band + width; no redundant par_lambda)
# --------------------------------------------------
out = xr.Dataset(
    {
        "par_band": par_band,
        "par_band_width": par_band_width
        # Optional: keep weights for debugging (uncomment if you want)
        # ,"weights": weights_da
    }
)

out.attrs.update({
    "title": "Spectral PAR forcing derived from ERA5 SSR using ROMS ec_Fobar weights",
    "source_file": infile,
    "note": "SSR is accumulated (J m-2). Converted to step flux via diff(time)/dt; PAR=0.43*SW; distributed spectrally with normalized ec_Fobar weights."
})

# --------------------------------------------------
# Checks (works on older xarray: use numpy abs)
# --------------------------------------------------
err = np.abs(out["par_band"].sum("wavelength") - PAR_total).max()
print("Max conservation error:", float(err))

max_night = float(out["par_band"].isel(time=0).max())
print("Max value at first timestep (should be ~0 at night):", max_night)

# --------------------------------------------------
# Write NetCDF
# --------------------------------------------------
out.to_netcdf(outfile)
print("Output written:", outfile)
