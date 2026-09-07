"""Explicit optional download; never invoked by preparation or the notebook."""
import argparse
import wfdb

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',choices=['svdb','incart'],required=True)
    args=p.parse_args();database='incartdb' if args.source=='incart' else 'svdb'
    print('Explicit download from https://physionet.org/content/'+database+'/1.0.0/')
    wfdb.dl_database(database,'data/'+args.source)
