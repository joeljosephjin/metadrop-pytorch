import os
from tqdm import tqdm
import requests


if not os.path.isdir('data'):
  os.makedirs('data')

def download_file(url, filename):
  chunkSize = 1024
  r = requests.get(url, stream=True)
  with open(filename, 'wb') as f:
    # pbar = tqdm( unit="B", total=int( r.headers['Content-Length'] ) )
    pbar = tqdm( unit="B", total=None )
    for chunk in r.iter_content(chunk_size=chunkSize):
      if chunk: # filter out keep-alive new chunks
        pbar.update (len(chunk))
        f.write(chunk)
  return filename


# Download the dataset
path = os.path.join('data', 'omniglot')
if not os.path.isdir(path):
  os.makedirs(path)

print("Downloading train.npy of Omniglot\n")
download_file('https://www.dropbox.com/s/h13g4b2awd7xdr6/train.npy?dl=1', os.path.join(path, 'train.npy'))
print("Downloading test.npy of Omniglot\n")
download_file('https://www.dropbox.com/s/w313ybz6rls1e83/test.npy?dl=1', os.path.join(path, 'test.npy'))
print("Downloading done.\n")
