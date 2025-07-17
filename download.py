import kagglehub
import argparse
def download_wich(setname):
    path = kagglehub.dataset_download(setname ="deadskull7/fer2013", path="~/.data")

    print("Path to dataset files:", path)
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setname",  required=True,
                        help="e.g. deadskull7/fer2013")
    args = parser.parse_args()
    download_wich(args.setname)