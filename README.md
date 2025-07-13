# AI_public_projects

💡 *A repo that contains all my public AI projects.*

*Dev Note:* Every sub project must have its own **.gitignore**



### How To Clone A Sub-Project

```bash
# Set PROJECT to the project you want to download, example: "object_detection/yolo_v1_taco" or "misc"
PROJECT=""
REPO_URL="https://github.com/t20e/AI_public_projects"
git clone --filter=blob:none --no-checkout "$REPO_URL"
cd AI_public_projects
git sparse-checkout init --cone
# You can also add more to set to download other projects.
git sparse-checkout set "$PROJECT"
git checkout main
echo "✅ Successfully extracted '$PROJECT' -> from -> $REPO_URL"
```



### 📌 Notable projects

<!-- TODO add taco back in after its been fixed -->
<!-- - 🔗 [yolo_v1_taco](./object_detection/yolo_v1_taco)
    - Object detection.
    - Vectorized.
    - [yolo_v1 resource](https://github.com/t20e/res/tree/main/coding.res/AI.res/object_detection/YOLO.res) -->

### Less-Notable projects

