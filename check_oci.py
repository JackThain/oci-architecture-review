"""Quick check that your OCI config file (outside this repo) is valid."""
import oci

config = oci.config.from_file()  # reads C:\Users\<you>\.oci\config
oci.config.validate_config(config)
print("Config looks good. Region:", config["region"])
