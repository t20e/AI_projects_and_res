# Utility Functions for bash scripts

 

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
        echo "Error: File '$zip_path' not found."
        return 1
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
                echo "❌ Failed to move contents from $DEEPLY_NESTED_PATH"
                exit 1
            }
            shopt -u dotglob # Disable dotglob after use
            echo "\n✅ Successfully moved contents from"
            echo "$DEEPLY_NESTED_PATH"
            echo "⬇        to      ⬇"
            echo "$d_path/$folder"
            # Remove the now empty deeply nested folder
            rmdir "$DEEPLY_NESTED_PATH" || {
                echo "❌ Failed to remove empty directory $DEEPLY_NESTED_PATH"
                exit 1
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