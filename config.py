ROOT = f'C:/Users/Asus/Desktop/Skoltech/Multiphase flows'
ROOT_SLASH = f'C:\\Users\\Asus\\Desktop\\Skoltech\\Multiphase flows'

# STUDY_NAME = 'Subsea_Study[1]'

DATA_PATH = f'{ROOT}/data'
# STUDY_PATH = f'{DATA_PATH}/{STUDY_NAME}'
# GEN_DFS_PATH = f'{STUDY_PATH}/gens'
# TPL_PATH = f'{ROOT_SLASH}\\olga_proj\\{STUDY_NAME}'

VOLVE_FILE = f'{DATA_PATH}/Volve production data_daily.xlsx'
REAL_FILE = f'{DATA_PATH}/real.csv'
REAL_FILE_ALL = f'{DATA_PATH}/real-all.csv'

TRAIN_VAL_RATIO = 8 / 10
TRAIN_VAL_RATIO_REAL = 1 / 3
TRAIN_VAL_RATIO_GEN = 8 / 10
N_STEPS = 5
N_FEATURES_IN = 6
N_FEATURES_OUT = 3