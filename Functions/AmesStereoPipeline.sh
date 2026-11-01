#!/usr/bin/env bash
set -Eeuo pipefail
shopt -s nullglob

# ============================================================
# Configuration
# ============================================================

# Pass any number of image prefixes, TIF files, or directories
# as arguments when running this script.
#
# Example:
# ./run.sh ./7091468262507664303-ord ./7092148974864472370-ord
#
# Or:
# ./run.sh path/to/image1 path/to/image2 path/to/image3.TIF

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <image_prefix|image.TIF|directory> [...]" >&2
    exit 1
fi

# Output/configuration
BA_DIR="./dg_csm_model_refined"
BA_PREFIX="${BA_DIR}/run"
DEM="/Users/chanagan/Downloads/Panama2026/VantorL1B/ASP/dem-adj.tif"

# Set these for your study area.
PROJ="+proj=utm +zone=17 +datum=WGS84 +units=m +no_defs"

# Projected map bounds: xmin ymax xmax ymin
# Replace with the appropriate bounds for your AOI.
T_XMIN="${T_XMIN:432310}" 
T_YMAX="${T_YMAX:861643}"
T_XMAX="${T_XMAX:623467}"
T_YMIN="${T_YMIN:801350}"

MAP_RES="${MAP_RES:-0.5}"
THREADS="${THREADS:-60}"
STEREO_PROCESSES="${STEREO_PROCESSES:-64}"

# ============================================================
# Discover input images
# ============================================================

declare -a INPUT_TIFS=()
declare -a PREFIXES=()
declare -a XMLS=()
declare -a CORRECTED=()
declare -a MAPPED=()

for arg in "$@"; do
    if [[ -d "$arg" ]]; then
        # Recursively find TIFFs in an order directory.
        while IFS= read -r -d '' tif; do
            INPUT_TIFS+=("$tif")
        done < <(
            find "$arg" -type f \
                \( -iname '*.tif' -o -iname '*.tiff' \) \
                -print0
        )
    elif [[ -f "$arg" ]]; then
        # Accept a TIF file directly.
        case "$arg" in
            *.TIF|*.tif|*.TIFF|*.tiff)
                INPUT_TIFS+=("$arg")
                ;;
            *)
                echo "Unsupported input file: $arg" >&2
                exit 1
                ;;
        esac
    else
        # Accept an image prefix without the extension.
        if [[ -f "${arg}.TIF" ]]; then
            INPUT_TIFS+=("${arg}.TIF")
        elif [[ -f "${arg}.tif" ]]; then
            INPUT_TIFS+=("${arg}.tif")
        else
            echo "Input not found: $arg" >&2
            exit 1
        fi
    fi
done

# Sort and remove duplicate image paths.
mapfile -t INPUT_TIFS < <(
    printf '%s\n' "${INPUT_TIFS[@]}" |
        sort -u
)

if [[ ${#INPUT_TIFS[@]} -lt 2 ]]; then
    echo "At least two images are required for stereo." >&2
    exit 1
fi

# ============================================================
# Validate inputs and build parallel arrays
# ============================================================

declare -A DATE_IMAGES=()

for tif in "${INPUT_TIFS[@]}"; do
    prefix="${tif%.*}"
    xml="${prefix}.XML"

    if [[ ! -f "$xml" ]]; then
        xml="${prefix}.xml"
    fi

    if [[ ! -f "$xml" ]]; then
        echo "Missing XML camera file for: $tif" >&2
        exit 1
    fi

    # Expected filename format:
    # 26OCT15160440-P1BS-200014610770_01_P001.TIF
    #
    # Capture DDMMMYY as the acquisition-date key.
    filename="$(basename "$prefix")"

    if [[ "$filename" =~ ^([0-9]{2}[A-Za-z]{3}[0-9]{2}) ]]; then
        date_key="${BASH_REMATCH[1]}"
        date_key="${date_key^^}"
    else
        echo "Cannot extract acquisition date from: $filename" >&2
        echo "Expected a filename beginning with DDMMMYY." >&2
        exit 1
    fi

    PREFIXES+=("$prefix")
    XMLS+=("$xml")
    CORRECTED+=("${prefix}.wv_correct")
    MAPPED+=("${prefix}.wv_correct.map")

    DATE_IMAGES["$date_key"]+="${#PREFIXES[@]} "
done

echo "Found ${#INPUT_TIFS[@]} input images."

# ============================================================
# Correct images
# ============================================================

for i in "${!PREFIXES[@]}"; do
    input="${INPUT_TIFS[$i]}"
    xml="${XMLS[$i]}"
    output="${CORRECTED[$i]}.TIF"

    if [[ ! -f "$output" ]]; then
        echo "Correcting: $input"
        wv_correct "$input" "$xml" "$output"
    else
        echo "Using existing corrected image: $output"
    fi
done

# ============================================================
# Bundle adjustment using all selected images
# ============================================================

mkdir -p "$BA_DIR"

declare -a BA_ARGS=()

# First: all corrected images.
for prefix in "${CORRECTED[@]}"; do
    BA_ARGS+=("${prefix}.TIF")
done

# Then: their corresponding XML camera files.
for xml in "${XMLS[@]}"; do
    BA_ARGS+=("$xml")
done

echo "Running bundle adjustment on ${#PREFIXES[@]} images..."

bundle_adjust \
    --ip-per-image 10000 \
    -t dg \
    --camera-weight 0 \
    --tri-weight 0.1 \
    --tri-robust-threshold 0.1 \
    "${BA_ARGS[@]}" \
    -o "$BA_PREFIX"


# ============================================================
# Mapproject all images
# ============================================================

for i in "${!PREFIXES[@]}"; do
    img="${CORRECTED[$i]}"
    xml="${XMLS[$i]}"
    output="${MAPPED[$i]}.TIF"

    # Skip if the mapprojected image already exists and is nonempty.
    if [[ -s "$output" ]]; then
        echo "Skipping existing mapprojected image: $output"
        continue
    fi

    echo "Mapprojecting: $img"

    mapproject \
        -t rpc \
        --threads "$THREADS" \
        --t_projwin \
            "$T_XMIN" "$T_YMAX" "$T_XMAX" "$T_YMIN" \
        --tr "$MAP_RES" \
        --t_srs "$PROJ" \
        --bundle-adjust-prefix "$BA_PREFIX" \
        "$DEM" \
        "${img}.TIF" \
        "$xml" \
        "$output"
done

# ============================================================
# Stereo by acquisition date
# ============================================================

for date_key in $(printf '%s\n' "${!DATE_IMAGES[@]}" | sort); do

    # Convert the stored image indices into an array.
    read -r -a indices <<< "${DATE_IMAGES[$date_key]}"

    n="${#indices[@]}"

    if (( n < 2 )); then
        echo "Skipping $date_key: only one image found."
        continue
    fi

    echo "Date $date_key: $n images; processing stereo pairs."

    # If a date has more than two images, process every unique pair.
    for ((a = 0; a < n - 1; a++)); do
        for ((b = a + 1; b < n; b++)); do

            i=$((indices[a] - 1))
            j=$((indices[b] - 1))

            left="${MAPPED[$i]}.TIF"
            right="${MAPPED[$j]}.TIF"
            left_xml="${XMLS[$i]}"
            right_xml="${XMLS[$j]}"

            left_id="$(basename "${PREFIXES[$i]}")"
            right_id="$(basename "${PREFIXES[$j]}")"

            # Separate output directory for each pair.
            pair_dir="stereo_${date_key}/${left_id}__${right_id}"
            mkdir -p "$pair_dir"

            stereo_prefix="${pair_dir}/run"

            echo "Stereo pair:"
            echo "  Left:  $left"
            echo "  Right: $right"
            echo "  Output: $stereo_prefix"

            parallel_stereo \
                -t dg \
                --max-disp-spread 200 \
                --processes "$STEREO_PROCESSES" \
                --accept-provided-mapproj-dem \
                --alignment-method none \
                --stereo-algorithm asp_bm \
                --subpixel-mode 3 \
                --bundle-adjust-prefix "$BA_PREFIX" \
                "$left" \
                "$right" \
                "$left_xml" \
                "$right_xml" \
                "$DEM" \
                "$stereo_prefix"

            # Convert point cloud and left image to map-projected outputs.
            point2dem \
                --t_srs "$PROJ" \
                --tr "$MAP_RES" \
                --dem-hole-fill-len 100 \
                --orthoimage-hole-fill-len 100 \
                "${stereo_prefix}-PC.tif" \
                --orthoimage "${stereo_prefix}-L.tif"

        done
    done
done

echo "All processing complete."
