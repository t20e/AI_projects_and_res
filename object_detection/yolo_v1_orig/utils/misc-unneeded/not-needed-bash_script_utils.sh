

# ‼️‼️‼️ Bash script not needed.
#  It is a headache to use BASH or ZSH to create a complex setup pipeline that is portables across OSs. Instead I ended up using Python.

# Note: using python is easier to implement however you can not create a conda env with it, for example when the repo is downloaded and u call setup.py and lets say its job is to create the env and activate it but this will cause an error if setup.py has python modules that are not installed, we first what to create the env than activate it and this way when we use python the modules are already installed.


# Utility Functions for bash scripts
set -euo pipefail

#================================================================
# FUNCTION: progress_bar
# DESCRIPTION: Displays a simple progress bar in the terminal.
#================================================================
function progress_bar() {
    # Creates a progress bar that shows progress of unzipping files.
    local current_progress=$1
    local total=$2

    local bar_width=50

    # Avoid division by zero
    if [ "$total" -eq 0 ]; then
        total=1
    fi

    local percentage=$(( (current_progress * 100) / total ))
    local completed_width=$(( (current_progress * bar_width) / total ))

    # Create the bar
    local bar=""
    for ((i=0; i<completed_width; i++)); do bar+="#"; done
    for ((i=completed_width; i<bar_width; i++)); do bar+="-"; done

    printf "\r[%s] %d%% (%d/%d)" "$bar" "$percentage" "$current_progress" "$total"
}

#================================================================
# FUNCTION: unzip_and_monitor
# DESCRIPTION: Unzips a file and displays a progress bar.
#================================================================
function unzip_and_monitor() {
    
    local zip_path=$1 # The path to the zip file.
    local d_path=$2 # The target path where the contains of the zip file are being extracted to (./datasets).

    # Make sure zip file exists
    if [ ! -f "$zip_path" ]; then
        error_exit "Error: File '$zip_path' not found."
    fi

    # --- Step 1: Get the total file count using the reliable method ---
    # echo "Calculating total files in archive..."
    local total_files=$(unzip -Z -1 "$zip_path" | wc -l)

    if [ "$total_files" -eq 0 ]; then
        echo "Archive is empty."
        return 0
    fi
    
    echo "Extracting $total_files files -> to -> $d_path"

    # --- Step 2: Start unzip in the background ---
    unzip -o -q "$zip_path" -d "$d_path" &
    local unzip_pid=$!

#     # --- Step 3: Monitor the progress ---
    while kill -0 $unzip_pid 2>/dev/null; do
        current_files=$(find "$d_path" -type f | wc -l)
        
        if [ "$current_files" -gt "$total_files" ]; then
            current_files=$total_files
        fi

        progress_bar $current_files $total_files
        sleep 0.2
    done

    # --- Finalization ---
    progress_bar $total_files $total_files
    echo -e "\n\n✅ Extraction complete.\n"
}






#================================================================
# FUNCTION: remove_nested_folder
# DESCRIPTION: Removes unnecessary nested folders with the same name 
#                                               i.e. t/t -> /t
#================================================================

function remove_nested_folder() {
    local d_path=$1 # The dataset path (./datasets).
    local NESTED_FOLDERS=("VOC2012_train_val" "VOC2012_test")

    printf "Removing nested folders:\n\n"

    if [ -z "$NESTED_FOLDERS" ]; then
        echo "Error: No nested folders provided."
        return 1
    fi

    for folder in "${NESTED_FOLDERS[@]}"; do
        DEEPLY_NESTED_PATH="$d_path/$folder/$folder"

        # Check if the deep nested folder exists
        if [ -d "$DEEPLY_NESTED_PATH" ]; then
            echo "\n📁 Found nested structure: $DEEPLY_NESTED_PATH"

            # Move all contents from the deeply nested folders to its parent
            # The 'shopt -u dotglob' ensures hidden files are also moved 
            shopt -s dotglob
            mv "$DEEPLY_NESTED_PATH"/* "$d_path/$folder/" || {
                error_exit "❌ Failed to move contents from $DEEPLY_NESTED_PATH"
            }
            shopt -u dotglob # Disable dotglob after use
            echo "\n✅ Successfully moved contents from"
            echo "$DEEPLY_NESTED_PATH"
            echo "⬇        to      ⬇"
            echo "$d_path/$folder"
            # Remove the now empty deeply nested folder
            rmdir "$DEEPLY_NESTED_PATH" || {
                error_exit "❌ Failed to remove empty directory $DEEPLY_NESTED_PATH"
            }
        else
            echo "Skipping $folder: Deeply nested folder $DEEPLY_NESTED_PATH does not exist."
        fi
    done
}


#================================================================
# FUNCTION: print_header
# DESCRIPTION: Prints a header.
#================================================================
function print_header() {
    printf "\n%s\n" "----------------------------------------------------------------------"
    printf "                        %s\n" "$1"
    printf "%s\n\n" "----------------------------------------------------------------------"
}


#================================================================
# FUNCTION: check_dependencies
# DESCRIPTION: Checks if necessary dependencies are installed.
#================================================================
function check_dependencies() {
    print_header "Checking for Dependencies"

    printf "⚠️  Warning: If you're using a zsh terminal, please note: this project's bash scripts depend on specific packages (like unzip). If these packages are installed in your zsh environment but aren't accessible to bash, the project's scripts won't run, even if the commands work perfectly in your zsh terminal.\n\n"

    local dependencies=("curl" "unzip")
    
    for cmd in "${dependencies[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            error_exit "'$cmd' command not found. Please install it and ensure it's in your PATH."
        fi
        echo "✅ '$cmd' is available."
    done
}


#================================================================
# FUNCTION: split_train_val_sets
# DESCRIPTION: Split the train_val_set directory into train_set and val_set (80/20 split).
#================================================================
# function split_train_val_sets(){

#     set -x # Debug mode

#     local base_path=$1

#     printf "\n\nSplitting VOC train/validation set into training and validation sets (80/20) split...\n"

#     local percentage=0.20 # 20%

#     local source_dir="$base_path/VOC2012_train_val" 
#     local val_dir="$base_path/VOC2012_val"

#     # --- 1: Create the val directly.
#     # Create destination root if it doesn't exist
#     mkdir -p "$val_dir"

#     # ---- Function to move 20% of files/folders from source to destination
#     move_percentage() {
#         local src=$1
#         local dest=$2

#         # Ensure the destination directory exists
#         mkdir -p "$dest"

#         local total_files
#         total_files=$(find "$src" -maxdepth 1 -type f | wc -l)
#         total_files=${total_files##* } # Remove whitespace from wc output

#         if [ "$total_files" -eq 0 ]; then
#             printf "No files to move from %s\n" "$src"
#             # return
#         fi

#         # Get number of files to move.
#         local num_to_move
#         num_to_move=$(echo "$total_files * $percentage" | bc | cut -d. -f1)


#         if [ "$num_to_move" -eq 0 ] && [ "$total_files" -gt 0 ]; then
#             num_to_move=1 # return one if none found.
#         fi

#         printf "Moving %d of %d files from: %s \n ↓ to: ↓ \n %s\n" "$num_to_move" "$total_files" "$src" "$dest"
#         # Use find, head, and xargs to move files efficiently
#         while IFS= read -r file_to_move; do
#             mv -- "$file_to_move" "$dest/"
#         done < <(find "$src" -maxdepth 1 -type f | head -n "$num_to_move")
#     }

#     # Define a static array of directory names and loop through it.
#     local subdirs_to_process=( "JPEGImages" "Annotations")

#     for subdir_name in "${subdirs_to_process[@]}"; do
#         local current_src_dir="$source_dir/$subdir_name"
#         # # Make sure the source directory actually exists before processing
#         if [ -d "$current_src_dir" ]; then
#             printf "\nProcessing subdirectory: %s\n" "$subdir_name"
#             move_percentage "$current_src_dir" "$val_dir/$subdir_name"
#         else
#             error_exit "\nWarning: Subdirectory not found, skipping: %s\n" "$current_src_dir"
#         fi
#     done

#     # Todo rename VOC2012_train_val -> VOC2012_train
#     mv "$source_dir" "$base_path/VOC2012_train"
# }

#================================================================
# FUNCTION: split_train_val_sets
# DESCRIPTION: Split the train_val_set directory into train_set and val_set (80/20 split),
#              ensuring image and annotation files are moved as pairs.
#================================================================
function split_train_val_sets(){

    set -x # Uncomment for debug mode (verbose output)

    local base_path="$1" # e.g., /path/to/datasets

    printf "\n\nSplitting VOC train/validation set into training and validation sets (80/20 split), ensuring image-annotation pairs move together...\n"

    local val_percentage=0.20 # 20% for validation

    local source_base_dir="$base_path/VOC2012_train_val"
    local source_images_dir="$source_base_dir/JPEGImages"
    local source_annotations_dir="$source_base_dir/Annotations"

    local val_base_dir="$base_path/VOC2012_val"
    local val_images_dir="$val_base_dir/JPEGImages"
    local val_annotations_dir="$val_base_dir/Annotations"

    # --- 1: Create the destination directories for validation set.
    mkdir -p "$val_images_dir"
    mkdir -p "$val_annotations_dir"

    # --- 2: Get all image base names (e.g., "2007_000001") from the source directory.
    # We use these base names to decide which pairs to move.
    # This will help use move the pairs of annotations and images without moving only one to val dir and keep one in the train dir
    local all_image_basenames=()
    # Find .jpg files, extract just the filename without path, then remove the extension.
    while IFS= read -r f; do
        filename=$(basename -- "$f")
        basename_no_ext="${filename%.*}" # Removes the last '.' and everything after it
        all_image_basenames+=("$basename_no_ext")
    # JPG images only, if you have png adjust this.
    done < <(find "$source_images_dir" -maxdepth 1 -type f -name "*.jpg" | sort)

    # Total pairs of annotations and images.
    local total_pairs=${#all_image_basenames[@]}

    if [ "$total_pairs" -eq 0 ]; then
        error_exit "No image files found in %s to split. Exiting.\n" "$source_images_dir"
    fi

    # --- 3: Calculate the number of pairs to move to the validation set.
    local num_to_move
    num_to_move=$(echo "$total_pairs * $val_percentage" | bc | cut -d. -f1)

    # Ensure at least one pair is moved if total_pairs > 0 and num_to_move calculates to 0
    if [ "$num_to_move" -eq 0 ] && [ "$total_pairs" -gt 0 ]; then
        num_to_move=1
    fi

    printf "Preparing to move %d out of %d image-annotation pairs to validation set.\n" "$num_to_move" "$total_pairs"

    # --- 4: Randomly select 'num_to_move' basenames for the validation set.
    local selected_basenames_file=$(mktemp) # Create a temporary file to store selected basenames

    # Use 'shuf' for robust random selection
    if command -v shuf >/dev/null 2>&1; then
        printf "%s\n" "${all_image_basenames[@]}" | shuf | head -n "$num_to_move" > "$selected_basenames_file"
    else
        printf "Warning: 'shuf' command not found. Using a less efficient and less random fallback.\n"
        # Fallback for systems without 'shuf' (basic, not cryptographically secure)
        local indices_to_pick=()
        while [ "${#indices_to_pick[@]}" -lt "$num_to_move" ]; do
            rand_idx=$(( RANDOM % total_pairs ))
            # Check if this index has already been picked
            local already_picked=0
            for picked_idx in "${indices_to_pick[@]}"; do
                if [ "$picked_idx" -eq "$rand_idx" ]; then
                    already_picked=1
                    break
                fi
            done
            if [ "$already_picked" -eq 0 ]; then
                indices_to_pick+=("$rand_idx")
                printf "%s\n" "${all_image_basenames[rand_idx]}" >> "$selected_basenames_file"
            fi
        done
    fi


    # --- 5: Move the selected image-annotation pairs.
    local moved_count=0
    while IFS= read -r basename_to_move; do
        local img_file="$source_images_dir/${basename_to_move}.jpg"
        local anno_file="$source_annotations_dir/${basename_to_move}.xml"

        # Check if both files exist before attempting to move (important for data integrity)
        if [ -f "$img_file" ] && [ -f "$anno_file" ]; then
            mv -- "$img_file" "$val_images_dir/"
            mv -- "$anno_file" "$val_annotations_dir/"
            moved_count=$((moved_count + 1))
        else
            printf "Warning: Missing image or annotation file for basename '%s'. Image: %s, Annotation: %s. Skipping move.\n" "$basename_to_move" "$img_file" "$anno_file"
        fi
    done < "$selected_basenames_file"

    printf "Successfully moved %d image-annotation pairs to validation set.\n" "$moved_count"

    # --- Clean up temporary file
    rm "$selected_basenames_file"

    # --- 6: Rename the remaining source directory to be the training set.
    local final_train_dir="$base_path/VOC2012_train"
    if [ -d "$source_base_dir" ]; then # Ensure the source directory still exists
        mv "$source_base_dir" "$final_train_dir"
        printf "Renamed remaining source directory (%s) to training directory (%s).\n" "$source_base_dir" "$final_train_dir"
    else
        printf "Source directory %s is empty or already moved. Not renaming.\n" "$source_base_dir"
    fi

    printf "Dataset splitting complete.\n"
}


# Function to handle errors and exit
error_exit() {
    local error_message="$1"
    local exit_code="${2:-1}" # Default exit code to 1 if not provided
    echo "ERROR: $error_message" >&2
    # exit "$exit_code"
    exit 1
}