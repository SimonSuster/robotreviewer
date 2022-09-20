import argparse
import csv
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
import torch
from setuptools._vendor.more_itertools import divide
from torch import nn, optim

from robotreviewer.util import load_json

sys.path.append( '/home/ssuster/robotreviewer/' )
import spacy
nlp = spacy.load('en_core_web_sm')

# https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
from tqdm import tqdm

from robotreviewer.label_pdfs import get_list_of_pdfs, get_articles, prepare_articles
from robotreviewer.ml.classifier import sigmoid, sigmoid_torch
from robotreviewer.robots.prob_bias_robot import ProbBiasRobot
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TemperatureScaler(nn.Module):
    def __init__(self, model, pmid2gold_f):
        super(TemperatureScaler, self).__init__()
        self.model = model
        #self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self.temperatures = {
            'Random sequence generation': nn.Parameter(torch.ones(1, device=device) * 1.5),
            'Allocation concealment': nn.Parameter(torch.ones(1, device=device) * 1.5),
            'Blinding of participants and personnel': nn.Parameter(torch.ones(1, device=device) * 1.5),
            'Blinding of outcome assessment': nn.Parameter(torch.ones(1, device=device) * 1.5)
            }
        self.temperatures_cnn = {
            'Random sequence generation': nn.Parameter(torch.ones(1, device=device) * 1.5),
            'Allocation concealment': nn.Parameter(torch.ones(1, device=device) * 1.5),
            'Blinding of participants and personnel': nn.Parameter(torch.ones(1, device=device) * 1.5),
            'Blinding of outcome assessment': nn.Parameter(torch.ones(1, device=device) * 1.5)
        }
        self.pmid2gold = load_json(pmid2gold_f)

    def forward_svm_only(self, output):
        criteria = [i["domain"] for i in output]
        logits_for_domains = torch.tensor([output_per_domain["logits_linear"] for output_per_domain in output], device=device)
        scaled_logits = self.temperature_scale(logits_for_domains)

        scaled_prob_linear = sigmoid_torch(scaled_logits)
        scaled_bias_prob = scaled_prob_linear
        # if we have a CNN pred, too, then average; otherwise rely on linear model.
        #prob_CNN_for_domains = torch.tensor([output_per_domain["bias_prob_CNN"] for output_per_domain in output],
                                              #device=device)
#        scaled_bias_prob = (prob_CNN_for_domains + scaled_prob_linear) / 2.0

        return scaled_logits, scaled_bias_prob, criteria

    def forward_separate_temp(self, output):
        criteria = [i["domain"] for i in output]
        logits_for_domains = torch.tensor([output_per_domain["logits_linear"] for output_per_domain in output],
                                          device=device)
        probs_for_domains = torch.tensor([output_per_domain["bias_prob_CNN"] for output_per_domain in output],
                                          device=device)
        scaled_logits = self.temperature_scale(logits_for_domains)

        scaled_prob_linear = sigmoid_torch(scaled_logits)
        scaled_bias_prob = scaled_prob_linear
        # if we have a CNN pred, too, then average; otherwise rely on linear model.
        # prob_CNN_for_domains = torch.tensor([output_per_domain["bias_prob_CNN"] for output_per_domain in output],
        # device=device)
        #        scaled_bias_prob = (prob_CNN_for_domains + scaled_prob_linear) / 2.0

        return scaled_logits, scaled_bias_prob, criteria

    def forward(self, output):
        """
        temp-scaled SVM and CNN logits separately and criterion-wise
        """
        criteria = [i["domain"] for i in output]
        logits_for_domains = torch.tensor([output_per_domain["logits_linear"] for output_per_domain in output],
                                          device=device)
        logits_CNN_for_domains = torch.tensor([output_per_domain["logits_CNN"] for output_per_domain in output],
                                         device=device)
        scaled_logits = []
        scaled_logits_CNN = []
        for i, criterion in enumerate(criteria):
            scaled_logits.append(self.temperature_scale_criterion_wise(logits_for_domains[i], criterion=criterion))
            scaled_logits_CNN.append(self.temperature_scale_criterion_wise_cnn(logits_CNN_for_domains[i],
                                                               criterion=criterion))

        scaled_prob_linear = sigmoid_torch(torch.tensor(scaled_logits, device=device))
        scaled_prob_CNN = sigmoid_torch(torch.tensor(scaled_logits_CNN, device=device))
        scaled_bias_prob = (scaled_prob_CNN + scaled_prob_linear) / 2.0

        return scaled_logits, scaled_logits_CNN, scaled_bias_prob, criteria

    def forward_combined(self, output):
        criteria = [i["domain"] for i in output]
        # treating the combined RR's output prob as our logit
        logits_for_domains = torch.tensor([output_per_domain["prob"] for output_per_domain in output],
                                          device=device)
        #scaled_logits = self.temperature_scale(logits_for_domains)
        scaled_logits = []
        for i, criterion in enumerate(criteria):
            scaled_logits.append(self.temperature_scale_criterion_wise(logits_for_domains[i], criterion))

        scaled_logits = torch.tensor(scaled_logits, device=device)
        scaled_prob = sigmoid_torch(scaled_logits)

        return scaled_logits, scaled_prob, criteria

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits. The logits are one per bias domain.
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature
        return logits / temperature

    def temperature_scale_criterion_wise(self, logits, criterion):
        # Expand temperature to match the size of logits
        temperature = self.temperatures[criterion]
        return logits / temperature

    def temperature_scale_criterion_wise_cnn(self, logits, criterion):
        """maintain separate temp param for CNN models"""
        # Expand temperature to match the size of logits
        temperature = self.temperatures_cnn[criterion]
        return logits / temperature

    def predict(self, valid_loader):
        # First: collect all the logits and labels for the validation set
        logits_list = []
        logits_CNN_list = []
        labels_list = []
        criteria_list = []
        probs_list = []
        filename_list = []

        with torch.no_grad():
            for c, (article, pdffile) in enumerate(tqdm(zip(*valid_loader))):
                output = self.model.annotate(article, pdffile)  # includes logits_linear
                if output == -1:
                    continue
                # forward pass
                logits, logits_CNN, probs, criteria = self(output)  # logits from the multitask SVM, probs from combined SVM/CNN
                # probs_list.append(probs)
                pmid = os.path.splitext(os.path.basename(pdffile))[0]
                for i, criterion in enumerate(criteria):
                    gold = self.get_gold(pmid, criterion)
                    if gold is not None:
                        labels_list.append(gold)
                        criteria_list.append(criterion)
                        logits_list.append(logits[i])
                        logits_CNN_list.append(logits_CNN[i])
                        probs_list.append(probs[i])
                        filename_list.append(pdffile)
            pos_logits = torch.tensor(logits_list).cuda()
            pos_logits_CNN = torch.tensor(logits_CNN_list).cuda()

            neg_logits = 1 - pos_logits
            neg_logits_CNN = 1 - pos_logits_CNN

            logits = torch.cat((neg_logits.unsqueeze(1), pos_logits.unsqueeze(1)), 1)
            logits_CNN = torch.cat((neg_logits_CNN.unsqueeze(1), pos_logits_CNN.unsqueeze(1)), 1)
            labels = torch.tensor(labels_list).cuda()

        return logits, logits_CNN, labels, criteria_list, probs_list, filename_list

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        nll_criterion = nn.NLLLoss().cuda()

        logits, logits_CNN, labels, criteria_list, _, _ = self.predict(valid_loader)

        for criterion, temperature in self.temperatures.items():
            # Next: optimize the temperature w.r.t. NLL
            optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)
            optimizer_CNN = optim.LBFGS([self.temperatures_cnn[criterion]], lr=0.01, max_iter=50)
            logits_criterion = logits[np.array(criteria_list) == criterion]
            logits_CNN_criterion = logits_CNN[np.array(criteria_list) == criterion]

            if logits_criterion.nelement() == 0 or logits_CNN_criterion.nelement() == 0:
                continue
            labels_criterion = labels[np.array(criteria_list) == criterion]

            # Calculate NLL and ECE before temperature scaling
            before_temperature_nll = nll_criterion(logits_criterion, labels_criterion).item()
            print('Before temperature - %s NLL: %.3f' % (criterion, before_temperature_nll))
            before_temperature_nll = nll_criterion(logits_CNN_criterion, labels_criterion).item()
            print('Before temperature (CNN) - %s NLL: %.3f' % (criterion, before_temperature_nll))

            def eval():
                optimizer.zero_grad()
                loss = nll_criterion(self.temperature_scale_criterion_wise(logits_criterion, criterion), labels_criterion)
                loss.backward()
                return loss

            def eval_cnn():
                optimizer_CNN.zero_grad()
                loss_cnn = nll_criterion(self.temperature_scale_criterion_wise_cnn(logits_CNN_criterion, criterion),
                                     labels_criterion)
                loss_cnn.backward()
                return loss_cnn

            optimizer.step(eval)
            optimizer_CNN.step(eval_cnn)

            # Calculate NLL and ECE after temperature scaling
            after_temperature_nll = nll_criterion(self.temperature_scale_criterion_wise(logits_criterion, criterion), labels_criterion).item()
            after_temperature_nll_CNN = nll_criterion(self.temperature_scale_criterion_wise_cnn(logits_CNN_criterion, criterion), labels_criterion).item()

            #after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
            print('Optimal temperature: %s %.3f' % (criterion, temperature.item()))
            #print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))
            print('After temperature - %s NLL: %.3f' % (criterion, after_temperature_nll))

            print('Optimal temperature CNN: %s %.3f' % (criterion, self.temperatures_cnn[criterion].item()))
            # print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))
            print('After temperature CNN - %s NLL: %.3f' % (criterion, after_temperature_nll_CNN))

        return self

    def get_gold(self, pmid, criterion):
        full2abbr = {"Allocation concealment": "allo_conceal",
                     "Blinding of participants and personnel": "part_blinding",
                     "Blinding of outcome assessment": "outcome_blinding",
                     "Random sequence generation": "rand_seq_gen"}

        assert self.pmid2gold[pmid]
        # take first annotation (only important if there are many)
        # if no annotation for the criterion exists, we return None
        label = self.pmid2gold[pmid][0].get(full2abbr[criterion], None)
        if label is None:
            return None
        else:
            return {"Unclear risk": 0, "High risk": 0, "Low risk": 1}[label]


def tune_temperature(articles, pdffiles, pmid2gold_f):
    orig_model = ProbBiasRobot()

    model = TemperatureScaler(orig_model, pmid2gold_f)
    model = model.to(device)

    # Tune the model temperature, and save the results
    model.set_temperature((articles, pdffiles))
    #output_model_f = os.path.join(serialization_dir, output_name)
    #torch.save(model.state_dict(), output_model_f)
    #print('Temperature scaled model saved to %s' % output_model_f)

    return model


def predict(articles, pdffiles, model):
    logits, logits_CNN, labels, criteria_list, probs_list, filename_list = model.predict((articles, pdffiles))

    return logits, logits_CNN, labels, criteria_list, probs_list, filename_list


def write(criteria_list, filename_list, probs_list, outfile_generic, write_header=True):
    assert len(criteria_list)  == len(filename_list) == len(probs_list)
    with open(f"{outfile_generic}.csv", "a") as fh_out:
        writer = csv.writer(fh_out)
        if write_header:
            writer.writerow(["domain", "filename", "prob"])
        for criterion, filename, prob in zip(criteria_list, filename_list, probs_list):
            writer.writerow([criterion, filename, prob.item()])


def main():
    # argument parser
    parser = argparse.ArgumentParser( description = 'Determine probabilistic bias in clinical trial pdfs' )
    parser.add_argument( '-i','--indir', help = 'Input directory - all pdfs will be parsed', required = True )
    parser.add_argument( '-o','--outfile_generic', help = 'Output csv file with pdf names and corresponding results', required = True )
    parser.add_argument('-pmid2gold_f', default="/home/ssuster/robotreviewer/robotreviewer_eval_data.json")
    parsed = parser.parse_args()
    if os.path.exists(f"{parsed.outfile_generic}.csv"):
        sys.exit(f"File {f'{parsed.outfile_generic}.csv'} already exists! Remove it first.")

    # get input/output info
    indir = parsed.indir

    # get list of pdfs
    print( '*** Get list of pdfs ***' )
    all_pdffiles = get_list_of_pdfs( indir )
    np.random.shuffle(all_pdffiles)

    n_folds = 2
    fold_0, fold_1 = divide(n_folds, all_pdffiles)
    folds = ([i for i in fold_0], [i for i in fold_1])

    runs = [folds, list(reversed(folds))]
    dbg_mode = False

    for n_run, run in enumerate(runs):
        dev_fold = run[0]
        test_fold = run[1]

        dev_pdffiles = [f for f in dev_fold]
        test_pdffiles = [f for f in test_fold]
        if dbg_mode:
            dev_pdffiles = dev_pdffiles[:5]
            test_pdffiles = test_pdffiles[:5]

        # tune temperature on the dev fold:
        # get articles
        print( '*** Get articles ***' )
        dev_articles = get_articles(dev_pdffiles)

        # prepare with tokenizer etc.
        print( '*** Prepare articles ***' )
        dev_prep_articles = prepare_articles(dev_articles)

        # classify articles
        print( '*** Classify articles ***' )
        model = tune_temperature(dev_prep_articles, dev_pdffiles, parsed.pmid2gold_f)

        # predict with tuned temp parameter on the test fold:
        # get articles
        print('*** Get articles ***')
        test_articles = get_articles(test_pdffiles)

        # prepare with tokenizer etc.
        print('*** Prepare articles ***')
        test_prep_articles = prepare_articles(test_articles)
        logits, logits_CNN, labels, criteria_list, probs_list, filename_list = predict(test_prep_articles, test_pdffiles, model)

        # write data.frame to csv
        print( '*** Write csv output ***' )
        write(criteria_list, filename_list, probs_list, parsed.outfile_generic, write_header=True if n_run==0 else False)


if __name__== "__main__":
    main()
