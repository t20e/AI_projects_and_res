# AI_public_projects

💡 *A repo that contains all my public AI projects.*

***Dev Note**: Every sub project must have its own .gitignore*



### How To Only Download A Sub-Project

Unfortunately Github makes cloning a sub-project difficult, the workaround is to use Github's `github.dev` web-based editor.

Steps:
1. Press the `period` key -> which will open it in github.dev editor.
2. Right-click on the sub-project you want and hit the download option.
3. Open that project in your IDE and read its `README.md`

<!-- - In a CLI (command-line_interface).

```shell
# Set PROJECT to the path of the project you want to download
    # Example: "object_detection/yolo_v1_taco" or "misc"
    PROJECT=""

    REPO_URL="https://github.com/t20e/AI_public_projects"
    git clone --filter=blob:none --no-checkout "$REPO_URL"

    cd AI_public_projects

    git sparse-checkout init --cone
    git sparse-checkout set "$PROJECT"
    # Note: You can also add more projects to set to download a many projects.
    git checkout main
    echo "✅ Successfully extracted '$PROJECT' -> from -> $REPO_URL"

# Open the project in your IDE.
    code "$PROJECT" 
# (cd) to that project.
    cd "$PROJECT"
# 2. ⭐️ Read that projects README.md
    code README.md

```  -->

### 📌 Notable projects

**Object detection.**

- 🔗 [yolo_v1_orig](https://github.com/t20e/AI_public_projects/tree/main/object_detection/yolo_v1_orig)
    - Implements the YOLO v1 paper.
    - [YOLOv1 notes](https://github.com/t20e/res/tree/main/coding.res/AI.res/object_detection/YOLO_v1.res)


#### Less-Notable projects

