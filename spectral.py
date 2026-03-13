import xarray as xr 
import numpy as np 



ds = xr.open_dataset("era5_sw_2024_01_german_baltic_smallbox.nc")
ds = ds.rename({"valid_time": "time"})
ssr = ds["ssr"]    #J m-2 (accumulated over 1 hour)
dt =3600.0
SW =ssr/dt        # W m-2


print("SW min/max:", float(SW.min()), float(SW.max()))

f_par = 0.43
PAR_total = f_par * SW
print("PAR_total min/max:", float(PAR_total.min()), float(PAR_total.max()))


wavelength = np.arange(402, 702, 5, dtype=float)  # 402..697
print("len(wavelength)=", len(wavelength), "first/last:", wavelength[0], wavelength[-1])


print(par_band.time.values[:5])
print(par_band.time.dtype)
print(par_band.sizes["time"])


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

assert ec_Fobar.size == 60, f"Expected 60 ec_Fobar values, got {ec_Fobar.size}"

w = ec_Fobar / ec_Fobar.sum()
print("len(w)=", w.size, "sum(w)=", float(w.sum()), "min/max w:", float(w.min()), float(w.max()))







# ---------- PLOT: one-day spectra + PAR time series (save as PNG) ----------
import matplotlib
matplotlib.use("Agg")  # IMPORTANT on HPC/servers (no GUI)

import matplotlib.pyplot as plt

# 1) Build spectral PAR per band (W m-2 per band)
par_band = PAR_total.expand_dims(wavelength=wavelength) * xr.DataArray(
    w, dims=["wavelength"], coords={"wavelength": wavelength}
)

# Optional: remove tiny numerical noise at night
par_band = par_band.where(PAR_total > 1e-6, 0.0)

PAR_total = par_band.sel(time=12).sum("wavelength")
print(PAR_total)





# 2) Choose one grid point (change indices if you like)
ilat, ilon = 2, 7
spec_point = par_band.isel(latitude=ilat, longitude=ilon)      # (valid_time, wavelength)
par_point  = PAR_total.isel(latitude=ilat, longitude=ilon)     # (valid_time)

# 3) Select one day (first 24 hours)
day_spec = spec_point.isel(valid_time=slice(0, 24))
day_par  = par_point.isel(valid_time=slice(0, 24))

# 4) Plot multiple spectra (several hours of the day)
plt.figure(figsize=(8, 5))
for t in [0, 6, 9, 12, 15, 18, 21]:
    plt.plot(wavelength, day_spec.isel(valid_time=t).values, label=f"hour {t}")

plt.xlabel("Wavelength (nm)")
plt.ylabel("PAR per band (W m-2)")
plt.title("Spectral PAR (Day 1) at one grid point")
plt.legend()
plt.tight_layout()
plt.savefig("spectral_day1.png", dpi=200)
plt.close()
print("Saved: spectral_day1.png")

# 5) Plot PAR_total time series for the same day
plt.figure(figsize=(8, 4))
plt.plot(np.arange(24), day_par.values)
plt.xlabel("Hour of day (0-23)")
plt.ylabel("Total PAR (W m-2)")
plt.title("Total PAR (Day 1) at same grid point")
plt.tight_layout()
plt.savefig("par_total_day1.png", dpi=200)
plt.close()
print("Saved: par_total_day1.png")

# 6) Energy conservation check: sum over wavelength = PAR_total
check = (par_band.sum("wavelength") - PAR_total).max().item()
print("Energy check (max error):", check)






