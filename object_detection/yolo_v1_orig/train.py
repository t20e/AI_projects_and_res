from configs.config_loader import load_config

cfg = load_config("yolov1.yaml", overrides=None)
print(cfg.CLASS_NAMES[1])