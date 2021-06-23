import pandas as pd
import os 
import logging
import traceback

def load_data(path: str):
    """
    Parameters:
        path : the path to the csv file (The file must be in cvs format)
    Returns:
        pd. DataFrame type
    """
    ext = os.path.splitext(path)[-1].lower()
    assert(ext == '.csv'), "The file must be in cvs format"
    try: 
        print(path)
        return pd.read_csv(path)
    except Exception as e:
        logging.error(traceback.format_exc())

