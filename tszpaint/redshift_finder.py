import asdf
from pathlib import Path
from config import HEALCOUNTS_PATH, DATA_PATH

def obtain_healcount_redshift(filepath):
    with asdf.open(filepath) as af:
        return af["headers"][0]["Redshift"]
    return redshift

def main():
    output_file = DATA_PATH / 'healcount_redshifts.txt'
    healcounts_dir = Path(HEALCOUNTS_PATH)
    
    with open(output_file, 'w') as f:
        for healcount_file in sorted(healcounts_dir.glob('LightCone0_halo_heal-counts_Step*.asdf')):
            try:
                redshift = obtain_healcount_redshift(healcount_file)
                f.write(f"{healcount_file.name} {redshift}\n")
            except Exception as e:
                print(f"Skipping {healcount_file.name}: {e}")

    
if __name__ == "__main__":
    main()       
        
