wget https://ot-data3.sdsc.edu/appRasterSelectService1771869415266196338619/rasters_COP30.tar.gz

unzip it

dem_geoid --geoid egm2008 --reverse-adjustment output_hh.tif -o dem

wv_correct ./200011916615_01_P003_PAN/22APR24220031-P1BS-200011916615_01_P003.TIF ./200011916615_01_P003_PAN/22APR24220031-P1BS-200011916615_01_P003.XML ./200011916615_01_P003_PAN/22APR24220031-P1BS-200011916615_01_P003.wv_correct.TIF
wv_correct ./200011916615_01_P009_PAN/22APR24220212-P1BS-200011916615_01_P009.TIF ./200011916615_01_P009_PAN/22APR24220212-P1BS-200011916615_01_P009.XML ./200011916615_01_P009_PAN/22APR24220212-P1BS-200011916615_01_P009.wv_correct.TIF

###################
#!/bin/bash
bundle_adjust --ip-per-image 10000 -t dg --camera-weight 0 --tri-weight 0.1 --tri-robust-threshold 0.1 \
    ./200011916615_01_P003_PAN/22APR24220031-P1BS-200011916615_01_P003.wv_correct.TIF \
    ./200011916615_01_P009_PAN/22APR24220212-P1BS-200011916615_01_P009.wv_correct.TIF \
    ./200011916615_01_P003_PAN/22APR24220031-P1BS-200011916615_01_P003.XML \
    ./200011916615_01_P009_PAN/22APR24220212-P1BS-200011916615_01_P009.XML \
    -o dg_csm_model_refined/run

# Set your projection 
proj="+proj=utm +zone=5 +datum=WGS84 +units=m +no_defs"

# List of prefixes
prefixes=("./200011916615_01_P003_PAN/22APR24220031-P1BS-200011916615_01_P003.wv_correct" \
    "./200011916615_01_P009_PAN/22APR24220212-P1BS-200011916615_01_P009.wv_correct")

prefixes2=("./200011916615_01_P003_PAN/22APR24220031-P1BS-200011916615_01_P003" \
    "./200011916615_01_P009_PAN/22APR24220212-P1BS-200011916615_01_P009")

# Loop over each prefix and run mapproject
# Loop over both arrays using index
for i in "${!prefixes[@]}"; do
  IMG="${prefixes[$i]}"
  XMLPREFIX="${prefixes2[$i]}"

  echo "Processing $IMG with XML from $XMLPREFIX..."

  mapproject -t rpc --threads 60 \
    --t_projwin 426848.05 6280562.76 439767.82 6288490.82 \ 
    --tr 0.5 \
    --t_srs "$proj" \
    --bundle-adjust-prefix dg_csm_model_refined/run \
    ./dem-adj.tif "${IMG}.TIF" "${XMLPREFIX}.XML" "${IMG}.map.TIF"
done


parallel_stereo -t dg --max-disp-spread 200 --processes 64 \
--accept-provided-mapproj-dem --alignment-method none \
--stereo-algorithm asp_bm  \
--subpixel-mode 3 \
--bundle-adjust-prefix dg_csm_model_refined/run \
./200011916615_01_P003_PAN/22APR24220031-P1BS-200011916615_01_P003.wv_correct.map.TIF \
./200011916615_01_P009_PAN/22APR24220212-P1BS-200011916615_01_P009.wv_correct.map.TIF \
./200011916615_01_P003_PAN/22APR24220031-P1BS-200011916615_01_P003.XML \
./200011916615_01_P009_PAN/22APR24220212-P1BS-200011916615_01_P009.XML \
run ./dem-adj.tif

proj="+proj=utm +zone=5 +datum=WGS84 +units=m +no_defs"

point2dem --t_srs "$proj" --tr 0.5 --dem-hole-fill-len 100 --orthoimage-hole-fill-len 100 run-PC.tif --orthoimage run-L.tif

######################
#!/bin/bash
#SBATCH --account=ehp
#SBATCH --partition=cpu
#SBATCH --time=01-00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chanagan@usgs.gov
#SBATCH --chdir=/caldera/hovenweep/projects/usgs/hazards/ehp/chanagan/Aiktalik

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

./run.sh 
#############################################################

gdalwarp -dstnodata -9999 -ot Float32 -tr 0.5 0.5 -te 532947.500 3151050.500 544017.500 3210468.000 -co COMPRESS=LZW -co ZLEVEL=9 -co BIGTIFF=YES -overwrite run2025W/run-2025WDRG.tif orthos/OrthoPostW-cropped.tif 
gdalwarp -dstnodata -9999 -ot Float32 -tr 0.5 0.5 -te 532947.500 3151050.500 544017.500 3210468.000 -co COMPRESS=LZW -co ZLEVEL=9 -co BIGTIFF=YES -overwrite run2022/run-2022DRG.tif orthos/OrthoPreW-cropped.tif 

gdalwarp -dstnodata -9999 -ot Float32 -tr 0.5 0.5 -te 547922.000 3151050.500 561092.500 3210471.500 -co COMPRESS=LZW -co ZLEVEL=9 -co BIGTIFF=YES -overwrite run2025E/run-2025EDRG.tif orthos/OrthoPostE-cropped.tif 
gdalwarp -dstnodata -9999 -ot Float32 -tr 0.5 0.5 -te 547922.000 3151050.500 561092.500 3210471.500 -co COMPRESS=LZW -co ZLEVEL=9 -co BIGTIFF=YES -overwrite run2024/run-2024DRG.tif orthos/OrthoPreE-cropped.tif 

gdalwarp -dstnodata -9999 -ot Float32 -tr 0.5 0.5 -te 541839.500 3151050.500 551823.500 3210469.000 -co COMPRESS=LZW -co ZLEVEL=9 -co BIGTIFF=YES -overwrite run2024/run-2024DRG.tif orthos/OrthoPreE-Olap-cropped.tif
gdalwarp -dstnodata -9999 -ot Float32 -tr 0.5 0.5 -te 541839.500 3151050.500 551823.500 3210469.000 -co COMPRESS=LZW -co ZLEVEL=9 -co BIGTIFF=YES -overwrite run2025W/run-2025WDRG.tif orthos/OrthoPostW-Olap-cropped.tif

gdalwarp -dstnodata -9999 -ot Float32 -tr 0.5 0.5 -te 532947.500 3151050.500 544017.500 3210468.000 -co COMPRESS=LZW -co ZLEVEL=9 -co BIGTIFF=YES -overwrite run2025W/run-2025WDEM.tif dems/DEMPostW-cropped.tif 
gdalwarp -dstnodata -9999 -ot Float32 -tr 0.5 0.5 -te 532947.500 3151050.500 544017.500 3210468.000 -co COMPRESS=LZW -co ZLEVEL=9 -co BIGTIFF=YES -overwrite run2022/run-2022DEM.tif dems/DEMPreW-cropped.tif 

gdalwarp -dstnodata -9999 -ot Float32 -tr 0.5 0.5 -te 547922.000 3151050.500 561092.500 3210471.500 -co COMPRESS=LZW -co ZLEVEL=9 -co BIGTIFF=YES -overwrite run2025E/run-2025EDEM.tif dems/DEMPostE-cropped.tif 
gdalwarp -dstnodata -9999 -ot Float32 -tr 0.5 0.5 -te 547922.000 3151050.500 561092.500 3210471.500 -co COMPRESS=LZW -co ZLEVEL=9 -co BIGTIFF=YES -overwrite run2024/run-2024DEM.tif dems/DEMPreE-cropped.tif 

gdalwarp -dstnodata -9999 -ot Float32 -tr 0.5 0.5 -te 541839.500 3151050.500 551823.500 3210469.000 -co COMPRESS=LZW -co ZLEVEL=9 -co BIGTIFF=YES -overwrite run2024/run-2024DEM.tif dems/DEMPreE-Olap-cropped.tif
gdalwarp -dstnodata -9999 -ot Float32 -tr 0.5 0.5 -te 541839.500 3151050.500 551823.500 3210469.000 -co COMPRESS=LZW -co ZLEVEL=9 -co BIGTIFF=YES -overwrite run2025W/run-2025WDEM.tif dems/DEMPostW-Olap-cropped.tif

#############################
mm3d Mm2dPosSism OrthoPreE-cropped.tif OrthoPostE-asp-aligned.tif CorMin=0.1 Dequant=false DirMEC='MEC_E/'  
mm3d Mm2dPosSism OrthoPreE-Olap-cropped.tif OrthoPostW-Olap-asp-aligned.tif CorMin=0.1 Dequant=false DirMEC='MEC_Olap/' 
mm3d Mm2dPosSism OrthoPreW-cropped.tif OrthoPostW-asp-aligned.tif CorMin=0.1 Dequant=false DirMEC='MEC_W/' 
