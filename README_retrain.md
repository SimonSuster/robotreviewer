
We prepare the data for retraining:

1) Collect the set of pmids that are in Cochrane CDSR (so we can get the reviewers' annotations). Although we could constrain the set of pmids to get to only those used in the development of RobotReviewer, we decided not to do this as it results in too few pmids.

See `/home/simon/Apps/SysRev/sysrev/dataset_construction/dataset/robotreviewer_train_prepare.py`.

2) Get the PMC portion of pdfs:

```bash
cd /home/simon/Apps/SysRevData/data/dataset/
bash robotreviewer_pdf_links
```

3) Scraping of the larger bulk of pdfs with PubMunch is done using pubMunch. This takes a few days. As we regularly encounter unicode-related errors during downloading, we set up a bash script to automatically restart the process when it fails.

```bash
cd ~/Apps/pubMunch
bash cron_script.sh
```

The script simply contains the following line:

```bash
while true; do ./pubCrawl2 -du cochrane/ && break; done
```

TODO Next, rename the scraped files and copy them to the location that will be read by RobotReviewer (this will merge the pubMunch-scraped pdfs with the PMC ones already in the destination directory):
```bash
cd ~/Apps/pubMunch/cochrane/files/
rename 's/.main././' *.main.pdf
find . -regex '\./.[0-9]+\.pdf' -exec cp '{}' ~/robotreviewer/rct_pdfs/ \;
```

To convert pdf files (still on doe):

```bash
screen -S robotreviewer
conda activate robotreviewer
export PYTHONPATH=/home/$MY_USERNAME/robotreviewer/:$PYTHONPATH
cd /home/$MY_USERNAME/robotreviewer/
python3.6 robotreviewer/convert_pdfs.py -i /home/$MY_USERNAME/robotreviewer/rct_pdfs/ -o /home/$MY_USERNAME/robotreviewer/rct_pdfs_converted/
``` 

TODO: create the RoB dataset with full texts.

An alternative to full texts is to use abstracts only. This results in a larger dataset.
We obtain the abstracts of the pmids found in Cochrane (i.e. for which we have RoB annotations).

```bash
cd /home/simon/Apps/robotreviewer/robotreviewer/
python create_abstract_data.py
-i
/home/simon/Apps/SysRevData/data/dataset/pmids_train.txt
-o
/home/simon/Apps/robotreviewer/pubmed_abstracts/
```

Then, create the RoB dataset (abstracts-only) with splits.

- Protected group: medical area
```bash
/home/simon/Apps/fairlib/venv/bin/python /home/simon/Apps/fairlib/data/src/RiskOfBias/create_dataset.py -input_dir /home/simon/Apps/robotreviewer/pubmed_abstracts/ -label_f /home/simon/Apps/SysRevData/data/dataset/robotreviewer_train_data.json -protected_label_f /home/simon/Apps/SysRevData/data/dataset/robotreviewer_topics_train.json -output_dir /home/simon/Apps/robotreviewer/rob_abstract_dataset_area/
```

- Protected group: age
```bash
/home/simon/Apps/fairlib/venv/bin/python /home/simon/Apps/fairlib/data/src/RiskOfBias/create_dataset.py -input_dir /home/simon/Apps/robotreviewer/pubmed_abstracts/ -label_f /home/simon/Apps/SysRevData/data/dataset/robotreviewer_train_data.json -protected_label_f /home/simon/Apps/SysRevData/data/dataset/robotreviewer_age_train.json -output_dir /home/simon/Apps/robotreviewer/rob_abstract_dataset_age/
```
- Protected group: sex
```bash
/home/simon/Apps/fairlib/venv/bin/python /home/simon/Apps/fairlib/data/src/RiskOfBias/create_dataset.py -input_dir /home/simon/Apps/robotreviewer/pubmed_abstracts/ -label_f /home/simon/Apps/SysRevData/data/dataset/robotreviewer_train_data.json -protected_label_f /home/simon/Apps/SysRevData/data/dataset/robotreviewer_sex_train.json -output_dir /home/simon/Apps/robotreviewer/rob_abstract_dataset_sex/
```