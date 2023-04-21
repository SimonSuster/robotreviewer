#!/usr/bin/env python

"""
Stand-alone script to process directory with pdfs to obtain txt
"""

import os
import sys
import time

from setuptools._vendor.more_itertools import chunked

from robotreviewer.robots.bias_robot import BiasRobot
from robotreviewer.util import read_csv, load_json, save_json

sys.path.append( '/home/ssuster/robotreviewer/' )

import spacy
nlp = spacy.load('en_core_web_sm')

import pandas as pd
import pickle
import argparse


from robotreviewer.robots.prob_bias_robot import ProbBiasRobot
from robotreviewer.textprocessing.pdfreader import PdfReader
from robotreviewer.textprocessing.tokenizer import nlp


#############################################

###
# Read pdf as binary file
##
def read_binary_pdf( pdffile ):
    f = open( pdffile, 'rb' )
    pdf_text = f.read()
    f.close()
    return( pdf_text )

###
# Get articles
##
def get_articles( pdffiles ):
    blobs = []
    for pdffile in pdffiles:
        blobs.append( read_binary_pdf( pdffile ) )

    # convert binary pdfs to articles
    pdf_reader = PdfReader()
    articles = pdf_reader.convert_batch( blobs )
    return( articles )

###
# Return list of pdfs
##
def get_list_of_pdfs( dir_name ):
	pdf_names = filter( lambda x: x.endswith( '.pdf' ), os.listdir( dir_name ) )
	pdf_files = [ dir_name + "/" + s for s in pdf_names ]
	return pdf_files

###
# Preparing text
##
def prepare_articles( articles ):
    parsed_articles = []
    out_articles = []
    for doc in nlp.pipe( ( d.get('text', u'') for d in articles ), batch_size = 1, n_threads = 6 ):
        parsed_articles.append( doc )

    for article, parsed_text in zip( articles, parsed_articles ):
        article._spacy['parsed_text'] = parsed_text
        out_articles.append( article )

    return( articles )

###
# Get bias probabilies
##
def classify_articles( articles, pdffiles ):
    # get probabilistic bias robot
    robot = ProbBiasRobot()
    # annotate and convert to data.frame
    df_all = pd.DataFrame()
    
    for article, pdffile in zip( articles, pdffiles ):
        print( pdffile )
        #df = pd.DataFrame( robot.annotate( article, pdffile ) )
        anno = robot.annotate(article, pdffile)
        if anno == -1:
            continue
        #anno = robot.pdf_annotate(article)
        df = pd.DataFrame(anno)
        df_all = pd.concat( [ df_all, df ] )

    return( df_all )


def join_batched_predictions(outfile_generic, pdffiles_chunked):
        # join the batched predictions
    for n in range(len(pdffiles_chunked)):
        if n == 0:
            from_row = 0
        else:
            from_row = 1
        with open(f"{outfile_generic}_{n}.csv") as fh, open(f"{outfile_generic}.csv", "a") as fh_out:
            for l in fh.readlines()[from_row:]:
                if l.strip():
                    fh_out.write(l)


def extract_txt_articles(articles):
    txt_articles = []
    abstract_articles = []
    for article in articles:
        txt_articles.append(article.data["grobid"]["text"])
        abstract_articles.append(article.data["grobid"]["abstract"])

    return txt_articles, abstract_articles


def to_output_format(pdffiles, txt_articles, abstract_articles, label_dict, protected_label_dict, output):
    assert len(pdffiles) == len(txt_articles) == len(abstract_articles)

    for f, t, a in zip(pdffiles, txt_articles, abstract_articles):
        pmid = os.path.splitext(os.path.basename(f))[0]
        if pmid not in label_dict:
            continue
        if pmid not in protected_label_dict:
            continue
        labels = label_dict[pmid][0]  # taking the first annotation when there are many
        output[pmid] = {"text": t, "abstract": a, "labels": labels, "protected_labels": protected_label_dict[pmid]}

    return output


def main():
    # argument parser
    parser = argparse.ArgumentParser( description = 'Determine probabilistic bias in clinical trial pdfs' )
    parser.add_argument( '-i','--indir', help = 'Input directory - all pdfs will be parsed', required = True )
    parser.add_argument( '-o','--outfile', help = 'Output file in json containing converted txt files and labels', required = True )
    parser.add_argument('-l', '--label_file', help='Json file containing labels for each pmid', required=True)
    parser.add_argument('-pl', '--protected_label_file', help='Json file containing protected labels for each pmid', required=True)
    parsed = parser.parse_args()

    # get input/output info
    indir = parsed.indir

    # get list of pdfs
    print( '*** Get list of pdfs ***' )
    all_pdffiles = get_list_of_pdfs( indir )

    pdffiles_chunked = list(chunked(all_pdffiles, 50))
    output = {}

    label_dict = load_json(parsed.label_file)
    protected_label_dict = load_json(parsed.protected_label_file)

    for n, pdffiles in enumerate(pdffiles_chunked):
        if n > 0:
            break

        # get articles
        print( '*** Get articles ***' )
        articles = get_articles(pdffiles)

        # extract text
        print('*** Extract text ***')
        txt_articles, abstract_articles = extract_txt_articles(articles)

        # update the output file with newly processed files
        output = to_output_format(pdffiles, txt_articles, abstract_articles, label_dict, protected_label_dict, output)
        print()

    print( '*** Write json output ***' )
    save_json(output, parsed.outfile)


# run main
if __name__== "__main__":
    main()

