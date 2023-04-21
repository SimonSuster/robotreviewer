import argparse
import os

import requests
import urllib3
from eutils import EutilsNCBIError
from lxml.etree import XMLSyntaxError
from metapub import PubMedFetcher
from metapub.exceptions import MetaPubError
from requests.exceptions import ChunkedEncodingError
from tqdm import tqdm

from robotreviewer.util import save_json


def main(input_f, output_dir):
    fetch = PubMedFetcher()

    if not os.path.exists(f"{output_dir}/abstracts"):
        os.makedirs(f"{output_dir}/abstracts")

    if not os.path.exists(f"{output_dir}/titles"):
        os.makedirs(f"{output_dir}/titles")

    # get the  PMIDs
    with open(input_f) as fh:
        pmids = [l.strip() for l in fh]

    # get stuff
    for pmid in tqdm(pmids):
        output_f_abstract = f"{output_dir}/abstracts/{pmid}.txt"
        output_f_title = f"{output_dir}/titles/{pmid}.txt"
        if os.path.exists(output_f_title) and os.path.exists(output_f_abstract):
            continue

        try:
            article = fetch.article_by_pmid(pmid)
        except (EutilsNCBIError, XMLSyntaxError, ChunkedEncodingError, requests.exceptions.ConnectionError, urllib3.exceptions.MaxRetryError, MetaPubError):
            continue
        title = article.title
        abstract = article.abstract
        #year = eval(article.year)

        if not abstract:
            continue

        with open(output_f_title, "w") as fh_out:
            fh_out.write(title)

        with open(output_f_abstract, "w") as fh_out:
            fh_out.write(abstract)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_f',
                        help='input file: pubmed ids')
    parser.add_argument('-output_dir',
                        help="directory containing all abstract files")
    args = parser.parse_args()

    main(args.input_f, args.output_dir)

