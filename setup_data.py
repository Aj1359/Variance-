import os
import urllib.request
import gzip
import shutil
import zipfile
import tarfile

DATASETS = {
    "Facebook": "https://snap.stanford.edu/data/facebook_combined.txt.gz",
    "ca-GrQc": "https://snap.stanford.edu/data/ca-GrQc.txt.gz",
    "ca-HepTh": "https://snap.stanford.edu/data/ca-HepTh.txt.gz",
    "Irvine": "https://snap.stanford.edu/data/CollegeMsg.txt.gz",
    "EU": "https://snap.stanford.edu/data/email-Eu-core.txt.gz",
    "Amazon": "https://snap.stanford.edu/data/bigdata/communities/com-amazon.ungraph.txt.gz",
    "GitHub": "https://snap.stanford.edu/data/git_web_ml.zip",
    "TwitchGamers": "https://snap.stanford.edu/data/twitch_gamers.zip",
    "Gemsec-Facebook": "https://snap.stanford.edu/data/gemsec_facebook_dataset.tar.gz",
    "Musae-Facebook": "https://raw.githubusercontent.com/benedekrozemberczki/MUSAE/master/input/edges/facebook_edges.csv",
    "Musae-Twitch": "https://snap.stanford.edu/data/twitch.zip",
    "Email-Enron": "https://snap.stanford.edu/data/email-Enron.txt.gz",
    
    "YouTube": "https://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz",       # ~1.1M nodes, 3M edges (Fast to run)
    "Pokec": "https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz",                         # ~1.6M nodes, 30M edges (Medium/Large)
    "Orkut": "https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz",           # ~3M nodes, 117M edges (Large)
    "LiveJournal": "https://snap.stanford.edu/data/bigdata/communities/com-lj.ungraph.txt.gz",        # ~4M nodes, 69M edges (Large)
}

OUT_DIR = "Social_Network"

def setup():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    for name, url in DATASETS.items():
        txt_path = os.path.join(OUT_DIR, f"{name}.txt")
        
        if os.path.exists(txt_path):
            print(f"[{name}] Ready: {txt_path}")
            continue
            
        print(f"Downloading {name}...")
        try:
            if url.endswith(".txt.gz"):
                gz_path = os.path.join(OUT_DIR, f"{name}.txt.gz")
                urllib.request.urlretrieve(url, gz_path)
                print(f"Extracting {name}...")
                with gzip.open(gz_path, 'rb') as f_in, open(txt_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                os.remove(gz_path)
                
            elif url.endswith(".zip"):
                zip_path = os.path.join(OUT_DIR, f"{name}.zip")
                urllib.request.urlretrieve(url, zip_path)
                print(f"Extracting {name}...")
                with zipfile.ZipFile(zip_path, 'r') as z:
                    if name == "Musae-Twitch":
                        csv_name = [n for n in z.namelist() if "ENGB_edges.csv" in n][0]
                    else:
                        csv_name = [n for n in z.namelist() if "edges.csv" in n][0]
                    with z.open(csv_name) as f_in, open(txt_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(zip_path)
                # Convert commas to spaces and remove CSV header if present
                convert_csv_to_edgelist(txt_path)
                
            elif url.endswith(".tar.gz"):
                tar_path = os.path.join(OUT_DIR, f"{name}.tar.gz")
                urllib.request.urlretrieve(url, tar_path)
                print(f"Extracting {name}...")
                with tarfile.open(tar_path, 'r:gz') as t:
                    csv_name = [n for n in t.getnames() if "company_edges.csv" in n][0]
                    with t.extractfile(csv_name) as f_in, open(txt_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(tar_path)
                # Convert commas to spaces and remove CSV header if present
                convert_csv_to_edgelist(txt_path)
                
            elif url.endswith(".csv"):
                csv_path = os.path.join(OUT_DIR, f"{name}.csv")
                urllib.request.urlretrieve(url, csv_path)
                os.rename(csv_path, txt_path)
                convert_csv_to_edgelist(txt_path)
                
            print(f"[{name}] Ready: {txt_path}")
        except Exception as e:
            print(f"  -> ERROR downloading/processing {name}: {e}")

def convert_csv_to_edgelist(filepath):
    """Replaces commas with spaces and removes headers to match expected edgelist format."""
    temp_path = filepath + ".tmp"
    with open(filepath, 'r', encoding='utf-8') as f_in, open(temp_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            # Skip CSV headers (e.g., 'id_1,id_2' or non-numeric starts)
            if not line[0].isdigit() and not line.startswith('#'):
                continue
            f_out.write(line.replace(',', ' ') + '\n')
    os.replace(temp_path, filepath)

if __name__ == "__main__":
    setup()
