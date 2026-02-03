using XGPaint
using Healpix
using NPZ
using Printf
using Statistics

println("="^60)
println("XGPAINT.JL COMPARISON")
println("="^60)

# Load catalog
catalog = npzread("../data/comparison_output/catalog.npz")
ra = catalog["ra"]
dec = catalog["dec"]
masses = catalog["masses"]
nside = Int(catalog["nside"])
python_time = haskey(catalog, "python_time") ? catalog["python_time"] : nothing

N_halos = length(masses)
println("Loaded $N_halos halos, NSIDE=$nside")

# Create model
model = XGPaint.Battaglia16ThermalSZProfile(
    Omega_c=0.2589,
    Omega_b=0.0486,
    h=0.6774
)

# Create map and workspace
res = Resolution(nside)
m = HealpixMap{Float64, RingOrder}(nside)
workspace = XGPaint.HealpixProfileWorkspace(nside, 0.1)  



# Paint
t0 = time()
XGPaint.paint!(m, workspace, model, masses, fill(0.5, N_halos), ra, dec)
t1 = time()

julia_time = t1 - t0

println("\n" * "="^60)
@printf("Julia painting completed in %.2f seconds\n", julia_time)
println("="^60)

# Save
output_file = "../data/comparison_output/julia_map.fits"
isfile(output_file) && rm(output_file)
Healpix.saveToFITS(m, output_file, typechar="D")

# Load Python map and compare
python_map = Healpix.readMapFromFITS("../data/comparison_output/python_map.fits", 1, Float64)

# Differences
abs_diff = abs.(m.pixels .- python_map.pixels)
mask = (m.pixels .> 0) .| (python_map.pixels .> 0)
#rel_diff = abs_diff ./ (abs.(python_map.pixels) .+ 1e-30)

println("\n" * "="^60)
println("COMPARISON: Python vs Julia")
println("="^60)

@printf("Absolute difference:\n")
@printf("  Mean:   %.6e\n", mean(abs_diff[mask]))
@printf("  Median: %.6e\n", median(abs_diff[mask]))
@printf("  Max:    %.6e\n", maximum(abs_diff))

# measure correlation: exclude zero pixels
correlation = cor(python_map.pixels[mask], m.pixels[mask])
@printf("\nCorrelation: %.6f\n", correlation)

total_py = sum(python_map.pixels)
total_jl = sum(m.pixels)
@printf("\nTotal signal:\n")
@printf("  Python: %.6e\n", total_py)
@printf("  Julia:  %.6e\n", total_jl)
@printf("  Diff:   %.2f%%\n", 100 * abs(total_jl - total_py) / total_py)

println("\nTiming comparison:")
@printf("Python (tszpaint):  %.2f seconds\n", python_time)
@printf("Julia (XGPaint):    %.2f seconds\n", julia_time)
speedup = python_time / julia_time
if speedup > 1.0
    @printf("Julia is %.2fx faster\n", speedup)
else
    @printf("Python is %.2fx faster\n", 1.0/speedup)
end

println("Files saved:")
println("  ../data/comparison_output/python_map.fits")
println("  ../data/comparison_output/julia_map.fits")
