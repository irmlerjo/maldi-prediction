import numpy as np

US_IMBLEARN = 'imblearn_undersampled'
US_RANDOM = 'random_undersampled'
US_NO = 'not_undersampled'

SPECIES_ECOLI = 'Escherichia coli'
SPECIES_SAUREUS = 'Staphylococcus aureus'
SPECIES_KLEBSIELLA = 'Klebsiella pneumoniae'
SPECIES_EPIDERMIS = 'Staphylococcus epidermis'

ANTIBIOTIC_CIPROFLOXACIN ='Ciprofloxacin' # all
ANTIBIOTIC_CEFTRIAXONE = 'Ceftriaxone' # no saureus
ANTIBIOTIC_CEFEPIME ='Cefepime'# no saureus
ANTIBIOTIC_PIPTAZ = 'Piperacillin-Tazobactam' # only ecoli
ANTIBIOTIC_TOBRAMYCIN ='Tobramycin'# no saureus
ANTIBIOTIC_FUSIDICACID ='Fusidic acid' # only saureus
ANTIBIOTIC_OXACILLIN ='Oxacillin' # only saureus
ANTIBIOTIC_MEROPENEM ='Meropenem' # only kpneu
ANTIBIOTIC_VANCOMYCIN = 'Vancomycin'


BINNING_6K = '6k'
BINNING_18K = '18k'
BINNING_18K_RAW = 'RAW'

DATASET_DRIAMSA ='DRIAMS_A'
DATASET_DRIAMSB ='DRIAMS_B'
DATASET_ALL ='DRIAMS_ABCD'
DATASET_SPECIES_COMBINED ='DRIAMS_A_ALL_SPECIES'

# Random State for algorithms that have some randomness to make it reproducable use it
RANDOM_STATE=42

METHOD_TREE = 'tree'
METHOD_RFO = 'rfo'
METHOD_DUMMY = 'dummy'
METHOD_LR = 'logreg'
METHOD_MLP = 'mlp'
METHOD_CNN = 'cnn'

IN_FEATURES = 'in_features'
LEARNING_RATE='learning_rate'
EPOCHS = 'epochs'
WEIGHTED_FLAG = 'weighted_flag'
BATCH_SIZE = 'batch_size'


# Evaluation
PREDICTIONS = 'preds'
PROBABILITIES = 'probas'
BEST_PARAMS = 'best_params'
LOSSES = 'losses'
BEST_PREDS ='best_preds'
BEST_PROBAS ='best_probas'


### Grid Search Parameters for Baseline Methods

PARAM_GRID_TREE = {
  'criterion': ['gini','entropy'],
  'class_weight': ['balanced', None],
  'max_depth': [8,32,128,512,None],
  'max_features': ['sqrt','log2',None]

}
PARAM_GRID_LR = {
        'C': 10.0 ** np.arange(-3, 4),  # 10^{-3}..10^{3} (10^-1...10^4)
        'solver':['liblinear','saga'],
        'penalty': ['l1', 'l2'],
        'class_weight':['balanced',None]
}
PARAM_GRID_RFO = {
          'criterion': ['gini', 'entropy'],
          'n_estimators': [10,100,500,1000],
          'max_features': ['sqrt', 'log2'],
          'class_weight': ['balanced', None]
      }
PARAM_GRID_DUMMY = {
    'strategy': ['uniform','stratified','prior','most_frequent']
}

################## DEEP LEARNING CONSTS #####################
OUTPUT_DIM = 1 



''' All Antibiotics and their categorization inside of DRIAMS
# antibiotic categorization
ab_cat_map = {'5-Fluorocytosine': 'ANTIMYCOTICS FOR SYSTEMIC USE',
              'Amikacin': 'AMINOGLYCOSIDE ANTIBACTERIALS',
              'Aminoglycosides': 'AMINOGLYCOSIDE ANTIBACTERIALS',
              'Amoxicillin': 'BETA-LACTAM ANTIBACTERIALS, PENICILLINS',
              'Amoxicillin-Clavulanic acid': 'BETA-LACTAM ANTIBACTERIALS, PENICILLINS',
              'Amoxicillin-Clavulanic acid_uncomplicated_HWI': 'BETA-LACTAM ANTIBACTERIALS, PENICILLINS',
              'Amphotericin B': 'ANTIMYCOTICS FOR SYSTEMIC USE',
              'Ampicillin-Amoxicillin': 'BETA-LACTAM ANTIBACTERIALS, PENICILLINS',
              'Anidulafungin': 'ANTIMYCOTICS FOR SYSTEMIC USE',
              'Azithromycin': 'MACROLIDES, LINCOSAMIDES AND STREPTOGRAMINS',
              'Aztreonam': 'OTHER BETA-LACTAM ANTIBACTERIALS',
              'Caspofungin': 'ANTIMYCOTICS FOR SYSTEMIC USE',
              'Cefazolin': 'OTHER BETA-LACTAM ANTIBACTERIALS',
              'Cefepime': 'OTHER BETA-LACTAM ANTIBACTERIALS',
              'Cefixime': 'OTHER BETA-LACTAM ANTIBACTERIALS',
              'Cefoxitin_screen': 'OTHER BETA-LACTAM ANTIBACTERIALS',
              'Cefpodoxime': 'OTHER BETA-LACTAM ANTIBACTERIALS',
              'Ceftazidime': 'OTHER BETA-LACTAM ANTIBACTERIALS',
              'Ceftriaxone': 'OTHER BETA-LACTAM ANTIBACTERIALS',
              'Cefuroxime': 'OTHER BETA-LACTAM ANTIBACTERIALS',
              'Chloramphenicol': 'AMPHENICOLS',
              'Ciprofloxacin': 'QUINOLONE ANTIBACTERIALS',
              'Clarithromycin': 'MACROLIDES, LINCOSAMIDES AND STREPTOGRAMINS',
              'Clindamycin': 'MACROLIDES, LINCOSAMIDES AND STREPTOGRAMINS',
              'Colistin': 'OTHER ANTIBACTERIALS',
              'Cotrimoxazole': 'SULFONAMIDES AND TRIMETHOPRIM',
              'Daptomycin': 'OTHER ANTIBACTERIALS',
              'Doxycycline': 'TETRACYCLINES',
              'Ertapenem': 'OTHER BETA-LACTAM ANTIBACTERIALS',
              'Erythromycin': 'MACROLIDES, LINCOSAMIDES AND STREPTOGRAMINS',
              'Fluconazole': 'ANTIMYCOTICS FOR SYSTEMIC USE',
              'Fosfomycin': 'OTHER ANTIBACTERIALS',
              'Fosfomycin-Trometamol': 'OTHER ANTIBACTERIALS',
              'Fusidic acid': 'OTHER ANTIBACTERIALS',
              'Gentamicin': 'AMINOGLYCOSIDE ANTIBACTERIALS',
              'Gentamicin_high_level': 'AMINOGLYCOSIDE ANTIBACTERIALS',
              'Imipenem': 'OTHER BETA-LACTAM ANTIBACTERIALS',
              'Itraconazole': 'ANTIMYCOTICS FOR SYSTEMIC USE',
              'Levofloxacin': 'QUINOLONE ANTIBACTERIALS',
              'Linezolid': 'OTHER ANTIBACTERIALS',
              'Meropenem': 'OTHER BETA-LACTAM ANTIBACTERIALS',
              'Meropenem_with_meningitis': 'OTHER BETA-LACTAM ANTIBACTERIALS',
              'Meropenem_with_pneumonia': 'OTHER BETA-LACTAM ANTIBACTERIALS',
              'Meropenem_without_meningitis': 'OTHER BETA-LACTAM ANTIBACTERIALS',
              'Metronidazole': 'OTHER ANTIBACTERIALS',
              'Micafungin': 'ANTIMYCOTICS FOR SYSTEMIC USE',
              'Minocycline': 'TETRACYCLINES',
              'Moxifloxacin': 'QUINOLONE ANTIBACTERIALS',
              'Mupirocin': 'ANTIBIOTICS FOR TOPICAL USE',
              'Nitrofurantoin': 'OTHER ANTIBACTERIALS',
              'Norfloxacin': 'QUINOLONE ANTIBACTERIALS',
              'Oxacillin': 'BETA-LACTAM ANTIBACTERIALS, PENICILLINS',
              'Penicillin': 'BETA-LACTAM ANTIBACTERIALS, PENICILLINS',
              'Penicillin_with_endokarditis': 'BETA-LACTAM ANTIBACTERIALS, PENICILLINS',
              'Penicillin_with_meningitis': 'BETA-LACTAM ANTIBACTERIALS, PENICILLINS',
              'Penicillin_with_other_infections': 'BETA-LACTAM ANTIBACTERIALS, PENICILLINS',
              'Penicillin_with_pneumonia': 'BETA-LACTAM ANTIBACTERIALS, PENICILLINS',
              'Penicillin_without_endokarditis': 'BETA-LACTAM ANTIBACTERIALS, PENICILLINS',
              'Piperacillin-Tazobactam': 'BETA-LACTAM ANTIBACTERIALS, PENICILLINS',
              'Posaconazole': 'ANTIMYCOTICS FOR SYSTEMIC USE',
              'Quinolones': 'QUINOLONE ANTIBACTERIALS',
              'Rifampicin': 'DRUGS FOR TREATMENT OF TUBERCULOSIS',
              'Rifampicin_1mg-l': 'DRUGS FOR TREATMENT OF TUBERCULOSIS',
              'Teicoplanin': 'OTHER ANTIBACTERIALS',
              'Teicoplanin_GRD': 'OTHER ANTIBACTERIALS',
              'Tetracycline': 'TETRACYCLINES',
              'Tigecycline': 'TETRACYCLINES',
              'Tobramycin': 'AMINOGLYCOSIDE ANTIBACTERIALS',
              'Vancomycin': 'OTHER ANTIBACTERIALS',
              'Vancomycin_GRD': 'OTHER ANTIBACTERIALS',
              'Voriconazole': 'ANTIMYCOTICS FOR SYSTEMIC USE',
              }


# antibiotic naming
ab_name_map = {
    'AN-Amikacin': 'Amikacin',
    'Amikacin': 'Amikacin',
    'Amikacin 01 mg/l': 'Amikacin_1mg-l',
    'Amikacin 04 mg/l': 'Amikacin_4mg-l',
    'Amikacin 20 mg/l': 'Amikacin_20mg-l',
    'Aminoglykoside': 'Aminoglycosides',
    'Amoxicillin...Clavulansaeure.bei.unkompliziertem.HWI': 'Amoxicillin-Clavulanic acid_uncomplicated_HWI',
    'Amoxicillin-Clavulansaeure.unkompl.HWI': 'Amoxicillin-Clavulanic acid_uncomplicated_HWI',
    'Amoxicillin-Clavulan': 'Amoxicillin-Clavulanic acid',
    'AMC-Amoxicillin/Clavulans\xc3\xa4ure': 'Amoxicillin-Clavulanic acid',
    'Amoxicillin-Clavulansaeure': 'Amoxicillin-Clavulanic acid',
    'Amoxicillin/Clavulansäure': 'Amoxicillin-Clavulanic acid',
    'Amoxicillin...Clavulansaeure': 'Amoxicillin-Clavulanic acid',
    'Amoxicillin/Clavulansaeure': 'Amoxicillin-Clavulanic acid',
    'Amoxicillin': 'Amoxicillin',
    'AMX-Amoxicillin': 'Amoxicillin',
    'Ampicillin': 'Ampicillin',
    'AM-Ampicillin': 'Ampicillin',
    'P-Benzylpenicillin': 'Benzylpenicillin',
    'Benzylpenicillin': 'Benzylpenicillin',
    'Benzylpenicillin andere': 'Benzylpenicillin_others',
    'Benzylpenicillin bei Meningitis': 'Benzylpenicillin_with_meningitis',
    'Benzylpenicillin bei Pneumonie': 'Benzylpenicillin_with_pneumonia',
    'Amphotericin.B': 'Amphotericin B',
    'Amphothericin B': 'Amphotericin B',
    'Ampicillin...Amoxicillin': 'Ampicillin-Amoxicillin',
    'SAM-Ampicillin/Sulbactam': 'Ampicillin-Sulbactam',
    'Ampicillin...Sulbactam': 'Ampicillin-Sulbactam',
    'Anidulafungin': 'Anidulafungin',
    'Azithromycin': 'Azithromycin',
    'ATM-Aztreonam': 'Aztreonam',
    'Aztreonam': 'Aztreonam',
    'Bacitracin': 'Bacitracin',
    'Caspofungin': 'Caspofungin',
    'Cefalotin/Cefazolin': 'Cefalotin-Cefazolin',
    'Cefamandol': 'Cefamandole',
    'Cefazolin': 'Cefazolin',
    'FEP-Cefepim': 'Cefepime',
    'Cefepim': 'Cefepime',
    'Cefepime': 'Cefepime',
    'Cefepim.1': 'Cefepime',
    'Cefixim': 'Cefixime',
    'Cefoperazon-Sulbactam': 'Cefoperazon-Sulbactam',
    'Cefoperazon-Sulbacta': 'Cefoperazon-Sulbactam',
    'CTX-Cefotaxim': 'Cefotaxime',
    'Cefotaxim': 'Cefotaxime',
    'Cefoxitin Screening Staph': 'Cefoxitin_screen',
    'Cefoxitin.Screen': 'Cefoxitin_screen',
    'OXSF-Cefoxitin-Screen': 'Cefoxitin_screen',
    'FOX-Cefoxitin': 'Cefoxitin',
    'Cefoxitin': 'Cefoxitin',
    'CPD-Cefpodoxim': 'Cefpodoxime',
    'Cefpodoxim': 'Cefpodoxime',
    'Ceftarolin': 'Ceftarolin',
    'CAZ-Ceftazidim': 'Ceftazidime',
    'Ceftazidim.1': 'Ceftazidime',
    'Ceftazidim': 'Ceftazidime',
    'Ceftazidim.Avibactam': 'Ceftazidime-Avibactam',
    'Ceftazidim-Avibactam': 'Ceftazidime-Avibactam',
    'Ceftibuten': 'Ceftibuten',
    'Ceftobiprol': 'Ceftobiprole',
    'Ceftolozan...Tazobactam': 'Ceftolozane-Tazobactam',
    'Ceftolozan-Tazobacta': 'Ceftolozane-Tazobactam',
    'Ceftriaxon': 'Ceftriaxone',
    'CRO-Ceftriaxon': 'Ceftriaxone',
    'CXMA-Cefuroxim-Axetil': 'Cefuroxime',
    'Cefuroxim.Axetil': 'Cefuroxime',
    'Cefuroxim iv': 'Cefuroxime',
    'CXM-Cefuroxim': 'Cefuroxime',
    'Cefuroxim': 'Cefuroxime',
    'Cefuroxim oral': 'Cefuroxime',
    'Chinolone': 'Quinolones',
    'C-Chloramphenicol': 'Chloramphenicol',
    'Chloramphenicol': 'Chloramphenicol',
    'Ciprofloxacin': 'Ciprofloxacin',
    'CIP-Ciprofloxacin': 'Ciprofloxacin',
    'Clarithromycin': 'Clarithromycin',
    'Clarithromycin 04': 'Clarithromycin_4mg-l',
    'Clarithromycin 16': 'Clarithromycin_16mg-l',
    'Clarithromycin 32': 'Clarithromycin_32mg-l',
    'Clarithromycin 64': 'Clarithromycin_64mg-l',
    'Clindamycin': 'Clindamycin',
    'CM-Clindamycin': 'Clindamycin',
    'Clindamycin ind.': 'Clindamycin_induced',
    'ICR-Induzierbare Clindamycin Resistenz': 'Clindamycin_induced',
    'Clofazimin': 'Clofazimine',
    'Clofazimin 0.25 mg/l': 'Clofazimine_.25mg-l',
    'Clofazimin 0.5 mg/l': 'Clofazimine_.5mg-l',
    'Clofazimin 1.0 mg/l': 'Clofazimine_1mg-l',
    'Clofazimin 2.0 mg/l': 'Clofazimine_2mg-l',
    'Clofazimin 4.0 mg/l': 'Clofazimine_4mg-l',
    'Colistin': 'Colistin',
    'CS-Colistin': 'Colistin',
    'Cotrimoxazol': 'Cotrimoxazole',
    'Trimethoprim/Sulfamethoxazol': 'Cotrimoxazole',
    'SXT-Trimethoprim/Sulfamethoxazol': 'Cotrimoxazole',
    'Trimethoprim-Sulfame': 'Cotrimoxazole',
    'DAP-Daptomycin': 'Daptomycin',
    'Daptomycin': 'Daptomycin',
    'ESBL': 'ESBL',
    'Doxycyclin': 'Doxycycline',
    'Dummy': 'Dummy',
    'Ertapenem': 'Ertapenem',
    'ETP-Ertapenem': 'Ertapenem',
    'E-Erythromycin': 'Erythromycin',
    'Erythromycin': 'Erythromycin',
    'Ethambutol': 'Ethambutol',
    'Ethambutol 02.5': 'Ethambutol_2mg-l',
    'Ethambutol 05.0': 'Ethambutol_5mg-l',
    'Ethambutol.5.0.mg.l': 'Ethambutol_5mg-l',
    'Ethambutol 07.5': 'Ethambutol_7.5mg-l',
    'Ethambutol 12.5': 'Ethambutol_12.5mg-l',
    'Ethambutol 50': 'Ethambutol_50mg-l',
    'Fluconazol': 'Fluconazole',
    'Fosfomycin.Trometamol': 'Fosfomycin-Trometamol',
    'FOS-Fosfomycin': 'Fosfomycin',
    'Fosfomycin': 'Fosfomycin',
    'FA-Fusidins\xc3\xa4ure': 'Fusidic acid',
    'Fusidinsaeure': 'Fusidic acid',
    'Fusidins\xc3\xa4ure': 'Fusidic acid',
    'Fusidinsäure': 'Fusidic acid',
    'GHLR-High-Level-Resistenz gegen Gentamicin': 'Gentamicin_high_level',
    'Gentamicin High Level': 'Gentamicin_high_level',
    'Gentamicin.High.level': 'Gentamicin_high_level',
    'HLG-Gentamicin, High-Level (Synergie)': 'Gentamicin_high_level',
    'HLS-Streptomycin, High-Level (Synergie)': 'Streptomycin_high_level',
    'Gentamicin': 'Gentamicin',
    'GM-Gentamicin': 'Gentamicin',
    'Imipenem': 'Imipenem',
    'IPM-Imipenem': 'Imipenem',
    'Isavuconazol': 'Isavuconazole',
    'Isoniazid': 'Isoniazid',
    'Isoniazid.0.1.mg.l': 'Isoniazid_.1mg-l',
    'Isoniazid 0\t1 mg/l': 'Isoniazid_.1mg-l',
    'Isoniazid.0.4.mg.l': 'Isoniazid_.4mg-l',
    'Isoniazid 0\t4 mg/l': 'Isoniazid_.4mg-l',
    'Isoniazid 1.0 mg/l': 'Isoniazid_1mg-l',
    'Isoniazid 10  mg/l': 'Isoniazid_10mg-l',
    'Isoniazid 3.0 mg/l': 'Isoniazid_3mg-l',
    'Itraconazol': 'Itraconazole',
    'Ketoconazol': 'Ketoconazole',
    'LEV-Levofloxacin': 'Levofloxacin',
    'Levofloxacin': 'Levofloxacin',
    'LNZ-Linezolid': 'Linezolid',
    'Linezolid': 'Linezolid',
    'Linezolid 01 mg/l': 'Linezolid_1mg-l',
    'Linezolid 04 mg/l': 'Linezolid_4mg-l',
    'Linezolid 16 mg/l': 'Linezolid_16mg-l',
    'MRSA': 'MRSA',
    'Meropenem.bei.Meningitis': 'Meropenem_with_meningitis',
    'Meropenem.bei.Pneumonie': 'Meropenem_with_pneumonia',
    'Meropenem.ohne.Meningitis': 'Meropenem_without_meningitis',
    'Meropenem': 'Meropenem',
    'MEM-Meropenem': 'Meropenem',
    'Meropenem-Vaborbactam': 'Meropenem-Vaborbactam',
    'Meropenem-Vaborbacta': 'Meropenem-Vaborbactam',
    'Metronidazol': 'Metronidazole',
    'Miconazol': 'Miconazole',
    'Micafungin': 'Micafungin',
    'Minocyclin': 'Minocycline',
    'MXF-Moxifloxacin': 'Moxifloxacin',
    'Moxifloxacin': 'Moxifloxacin',
    'Moxifloxacin 0.5': 'Moxifloxacin_.5mg-l',
    'Moxifloxacin 02.5': 'Moxifloxacin_2.5mg-l',
    'Moxifloxacin 10': 'Moxifloxacin_10mg-l',
    'MUP-Mupirocin': 'Mupirocin',
    'Mupirocin': 'Mupirocin',
    'Nalidixinsaeure': 'Nalidixin acid',
    'Nitrofurantoin': 'Nitrofurantoin',
    'FT-Nitrofurantoin': 'Nitrofurantoin',
    'Norfloxacin': 'Norfloxacin',
    'NOR-Norfloxacin': 'Norfloxacin',
    'Novobiocin': 'Novobiocin',
    'Ofloxacin': 'Ofloxacin',
    'OFL-Ofloxacin': 'Ofloxacin',
    'Oxacillin': 'Oxacillin',
    'Oxa/Flucloxacil.': 'Oxacillin',
    'OX1-Oxacillin': 'Oxacillin',
    'Pefloxacin': 'Pefloxacin',
    'Penicillin.bei.anderen.Infekten': 'Penicillin_with_other_infections',
    'Penicillin.bei.Endokarditis': 'Penicillin_with_endokarditis',
    'Penicillin.bei.Meningitis': 'Penicillin_with_meningitis',
    'Penicillin.bei.Pneumonie': 'Penicillin_with_pneumonia',
    'Penicillin.ohne.Endokarditis': 'Penicillin_without_endokarditis',
    'Penicillin.ohne.Meningitis': 'Penicillin_without_meningitis',
    'Penicillin': 'Penicillin',
    'PIP-Piperacillin]': 'Piperacillin',
    'Piperacillin/Tazobactam': 'Piperacillin-Tazobactam',
    'TZP-Piperacillin/Tazobactam': 'Piperacillin-Tazobactam',
    'Piperacillin...Tazobactam': 'Piperacillin-Tazobactam',
    'Piperacillin-Tazobac': 'Piperacillin-Tazobactam',
    'PT-Pristinamycin': 'Pristinamycine',
    'Polymyxin.B': 'Polymyxin B',
    'Polymyxin B': 'Polymyxin B',
    'Posaconazol': 'Posaconazole',
    'Pyrazinamid.100.0.mg.l': 'Pyrazinamide',
    'Pyrazinamid 100\t0 mg': 'Pyrazinamide',
    'Pyrazinamid': 'Pyrazinamide',
    'QDA-Quinupristin/Dalfopristin': 'Quinupristin-Dalfopristin',
    'Quinupristin-Dalfopr': 'Quinupristin-Dalfopristin',
    'Rifabutin 0.1 mg/l': 'Rifabutin_.1mg-l',
    'Rifabutin 0.4 mg/l': 'Rifabutin_.4mg-l',
    'Rifabutin 2 mg/l': 'Rifabutin_2mg-l',
    'Rifampicin 01.0 mg/l': 'Rifampicin_1mg-l',
    'Rifampicin.1.0.mg.l': 'Rifampicin_1mg-l',
    'RA-Rifampicin': 'Rifampicin',
    'Rifampicin': 'Rifampicin',
    'Rifampicin 02.0 mg/l': 'Rifampicin_2mg-l',
    'Rifampicin 04 mg/l': 'Rifampicin_4mg-l',
    'Rifampicin 20 mg/l': 'Rifampicin_20mg-l',
    'SPX-Sparfloxacin': 'Sparfloxacin',
    'Roxithromycin': 'Roxithromycin',
    'Spectinomycin': 'Spectinomycin',
    'Streptomycin.1.0.mg.l': 'Streptomycin',
    'Streptomycin': 'Streptomycin',
    'Strepomycin High Level': 'Strepomycin_high_level',
    'Streptomycin.High.level': 'Strepomycin_high_level',
    'Teicoplanin.GRD': 'Teicoplanin_GRD',
    'Teicoplanin': 'Teicoplanin',
    'TEC-Teicoplanin': 'Teicoplanin',
    'Tetracyclin': 'Tetracycline',
    'TE-Tetracyclin': 'Tetracycline',
    'TIC-Ticarcillin': 'Ticarcillin',
    'TCC-Ticarcillin/Clavulans\xc3\xa4ure': 'Ticarcillin-Clavulan acid',
    'Ticarcillin...Clavulansaeure': 'Ticarcillin-Clavulan acid',
    'TEL-Telithromycin': 'Telithromycin',
    'Tigecyclin': 'Tigecycline',
    'TGC-Tigecycline': 'Tigecycline',
    'Tobramycin': 'Tobramycin',
    'TM-Tobramycin': 'Tobramycin',
    'Vancomycin.GRD': 'Vancomycin_GRD',
    'Vancomycin': 'Vancomycin',
    'VA-Vancomycin': 'Vancomycin',
    'Voriconazol': 'Voriconazole',
    'X5.Fluorocytosin': '5-Fluorocytosine',
    '5-Fluorocytosin': '5-Fluorocytosine',
}
'''
