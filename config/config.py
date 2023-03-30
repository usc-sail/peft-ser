import yaml

config = {
    "data_dir": {
        "crema_d":      "/media/data/public-data/SER/crema_d/CREMA-D/",
        "iemocap":      "/media/data/sail-data/iemocap/",
        "meld":         "/media/data/public-data/SER/meld/MELD.Raw/",
        "msp-podcast":  "/media/data/sail-data/MSP-podcast",
        "msp-improv":   "/media/data/sail-data/MSP-IMPROV/MSP-IMPROV/",
        "ravdess":      "/media/data/public-data/SER/Ravdess",
    },
    "project_dir":      "/media/data/projects/speech-privacy/trust-ser"
}

with open('config.yml', 'w') as outfile:
    yaml.dump(config, outfile, default_flow_style=False)