import re

def read_config_file(filename='config/flood_preprocessing.sh', verbose=False):
    """
    Load config file into python
    
    Args:
    filename str: Path to config file with multiple lines of "key='value'"
    verbose bool: If true, prints config parameters
    
    Returns:
        config dict(str:str): Dictionary with config key and value
    """
    # Load config file into python
    config_file = open(filename)
    config_txt = config_file.read()
    config_file.close()

    # Read every parameter definition line, searching for a key=value pattern. 
    config = dict()
    all_config_vars = [x.group() for x in re.finditer( r'(.*?)=\'(.*?)\'', config_txt)] 
    
    # Split each parameter definition into environment variable and value
    for config_var in all_config_vars:
        env_var, val = config_var.split('=')
        val = val.replace('\'', '')
        config[env_var] = val
    
    if verbose:
        print(*[f'{k}: {config[k]}' for k in config.keys()], sep = "\n")
    return config

