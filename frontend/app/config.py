"""Configuration of web site"""
import os

# Setup api url
BONSAI_API_URL = os.getenv("BONSAI_API_URL", "http://api:8000")
# where reference genomes are found
DATA_DIR = os.getenv("DATA_DIR", "/tmp/data")

# Session secret key
SECRET_KEY = b"not-so-secret"

# parameters for finding similar samples
SAMPLE_VIEW_SIMILARITY_LIMIT = 10
SAMPLE_VIEW_SIMILARITY_THRESHOLD = 0.9
SAMPLE_VIEW_TYPING_METHOD = "minhash"
SAMPLE_VIEW_CLUSTER_METHOD = "single"

# res classes
AMR_CLASS = {}

# Antibiotic classes
ANTIBIOTIC_CLASSES = {
    "Aminoglycosides": [
        "Amikacin",
        "Arbekacin",
        "Dibekacin",
        "Gentamicin",
        "Isepamicin",
        "Kanamycin",
        "Micronomicin",
        "Neomycin",
        "Netilmicin",
        "Plazomicin",
        "Ribostamycin",
        "Sisomicin",
        "Streptomycin",
        "Tobramycin",
    ],
    "Penicillins": [
        "Amoxicillin",
        "Ampicillin",
        "Azlocillin",
        "Bacampicillin",
        "Benzathine benzylpenicillin",
        "Benzylpenicillin",
        "Clometocillin",
        "Cloxacillin",
        "Dicloxacillin",
        "Flucloxacillin",
        "Mecillinam",
        "Mezlocillin",
        "Nafcillin",
        "Oxacillin",
        "Penamecillin",
        "Pheneticillin",
        "Phenoxymethylpenicillin",
        "Piperacillin",
        "Pivampicillin",
        "Pivmecillinam",
        "Procaine benzylpenicillin",
        "Sulbenicillin",
    ],
    "Beta-lactam + inhibitor": [
        "Amoxicillin+clavulanic Acid",
        "Ampicillin/sulbactam",
        "Sultamicillin",
        "Piperacillin+tazobactam",
    ],
    "Macrolides": [
        "Azithromycin",
        "Clarithromycin",
        "Dirithromycin",
        "Erythromycin",
        "Josamycin",
        "Lincomycin",
        "Midecamycin",
        "Oleandomycin",
        "Roxithromycin",
        "Spiramycin",
        "Telithromycin",
    ],
    "Monobactams": ["Aztreonam"],
    "Carbapenems": [
        "Biapenem",
        "Doripenem",
        "Ertapenem",
        "Imipenem",
        "Imipenem/cilastatin",
        "Meropenem",
        "Meropenem-vaborbactam",
        "Panipenem",
        "Tebipenem",
    ],
    "Carboxypenicillins": ["Carbenicillin", "Temocillin", "Ticarcillin"],
    "Cephalosporins": [
        "Cefcapene pivoxil",
        "Cefdinir",
        "Cefditoren pivoxil",
        "Cefetamet pivoxil",
        "Cefixime",
        "Cefmenoxime",
        "Cefodizime",
        "Cefoperazone",
        "Cefotaxime",
        "Cefpiramide",
        "Cefpodoxime proxetil",
        "Ceftazidime",
        "Ceftazidime-avibactam",
        "Cefteram pivoxil",
        "Ceftibuten",
        "Ceftizoxime",
        "Ceftriaxone",
        "Latamoxef",
        "Cefaclor",
        "Cefamandole",
        "Cefbuperazone",
        "Cefmetazole",
        "Cefminox",
        "Cefonicid",
        "Ceforanide",
        "Cefotetan",
        "Cefotiam",
        "Cefotiam hexetil",
        "Cefoxitin",
        "Cefprozil",
        "Cefuroxime",
        "Flomoxef",
        "Cefacetrile",
        "Cefadroxil",
        "Cefalexin",
        "Cefalotin",
        "Cefapirin",
        "Cefatrizine",
        "Cefazedone",
        "Cefazolin",
        "Cefradine",
        "Cefroxadine",
        "Ceftezole",
        "Cefepime",
        "Cefoselis",
        "Cefozopran",
        "Cefpirome",
        "Ceftaroline fosamil",
        "Ceftobiprole medocaril",
        "Ceftolozane-tazobactam",
    ],
    "Amphenicols": ["Chloramphenicol", "Thiamphenicol"],
    "Tetracyclines": [
        "Chlortetracycline",
        "Doxycycline",
        "Eravacycline",
        "Lymecycline",
        "Metacycline",
        "Minocycline (IV)",
        "Minocycline (oral)",
        "Omadacycline",
        "Oxytetracycline",
        "Tetracycline",
    ],
    "Fluoroquinolones": [
        "Ciprofloxacin",
        "Delafloxacin",
        "Enoxacin",
        "Fleroxacin",
        "Flumequine",
        "Garenoxacin",
        "Gatifloxacin",
        "Gemifloxacin",
        "Levofloxacin",
        "Lomefloxacin",
        "Moxifloxacin",
        "Norfloxacin",
        "Ofloxacin",
        "Pazufloxacin",
        "Pefloxacin",
        "Prulifloxacin",
        "Rufloxacin",
        "Sitafloxacin",
        "Sparfloxacin",
        "Tosufloxacin",
    ],
    "Lincosamides": ["Clindamycin"],
    "Phenol derivatives": ["Clofoctol"],
    "Polymyxins": ["Colistin", "Polymyxin B"],
    "Glycopeptides": [
        "Dalbavancin",
        "Oritavancin",
        "Teicoplanin",
        "Telavancin",
        "Vancomycin (IV)",
        "Vancomycin (oral)",
    ],
    "Streptogramins": ["Dalfopristin-quinupristin", "Pristinamycin"],
    "Lipopeptides": ["Daptomycin"],
    "Penems": ["Faropenem"],
    "Phosphonics": ["Fosfomycin (IV)", "Fosfomycin (oral)"],
    "Steroid antibacterials": ["Fusidic Acid"],
    "Oxazolidinones": ["Linezolid", "Tedizolid"],
    "Imidazoles": ["Metronidazole (IV)", "Metronidazole (oral)"],
    "Nitrofurantoin": ["Nitrofurantoin"],
    "Rifamycins": ["Rifabutin", "Rifampicin", "Rifamycin", "Rifaximin"],
    "Aminocyclitols": ["Spectinomycin"],
    "Combination of antibiotics": ["Spiramycin/metronidazole"],
    "Trimethoprim - sulfonamide combinations": [
        "Sulfadiazine/trimethoprim",
        "Sulfamethizole/trimethoprim",
        "Sulfamethoxazole/trimethoprim",
        "Sulfametrole/trimethoprim",
        "Sulfamoxole/trimethoprim",
    ],
    "Glycylcyclines": ["Tigecycline"],
    "Trimethoprim": ["Trimethoprim"],
}
