
We prepare the data for the evaluation of RobotReviewer locally, see `/home/simon/Apps/SysRev/sysrev/dataset_construction/dataset/robotreviewer_eval_prepare.py`. The PMC Open Access Subset pdfs are scraped locally as well. Scraping of the larger bulk of pdfs with PubMunch is done on `$MY_USERNAME@$DOE_SERVER:/scratch/$MY_USERNAME/pubMunch/`. This takes a few days.  

Rename the scraped files (on doe) and copy them to the location that will be read by RobotReviewer (this will merge the pubMunch-scraped pdfs with the PMC ones already in the destination directory):
```bash
cd /scratch/$MY_USERNAME/pubMunch/cochrane/files/
rename 's/.main././' *.main.pdf
find . -regex '\./.[0-9]+\.pdf' -exec cp '{}' ~/robotreviewer/rct_pdfs/ \;
```

RobotReviewer is run on `$DOE_SERVER` by first following the preparation these steps:

```bash
screen -S rabbitmq
cd ~/rabbitmq_server-3.9.12
ERLANG_HOME=/home/$MY_USERNAME/erlang/
export PATH=$PATH:$ERLANG_HOME/bin
./sbin/rabbitmq-server
#rabbitmq-server

screen -S grobid
cd grobid-0.7.0/
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH=$PATH:$JAVA_HOME/bin
./gradlew run

screen -S celery
cd /home/$MY_USERNAME/robotreviewer/
export GIT_LFS=/home/$MY_USERNAME/gitlfs/
export PATH=$PATH:$GIT_LFS/bin
git-lfs pull
conda activate robotreviewer

celery -A robotreviewer.ml_worker worker --loglevel=info --concurrency=1 --pool=solo
```

To actually make predictions with RobotReviewer (still on doe):

```bash
screen -S robotreviewer
conda activate robotreviewer
export PYTHONPATH=/home/$MY_USERNAME/robotreviewer/:$PYTHONPATH
cd /home/$MY_USERNAME/robotreviewer/
python3.6 robotreviewer/label_pdfs.py -i /home/$MY_USERNAME/robotreviewer/rct_pdfs/ -o /home/$MY_USERNAME/robotreviewer/rct_pdfs_out
``` 

We analyse the results using `/home/simon/Apps/SysRev/sysrev/modelling/allennlp/analyse_robotreviewer.py`, which runs on `$SLUG_SERVER`. Before analysing the results, copy the output file to slug: 

```bash
scp $MY_USERNAME@$DOE_SERVER:/home/$MY_USERNAME/robotreviewer/rct_pdfs_out.csv $MY_USERNAME@$SLUG_SERVER:/home/$MY_USERNAME/tmp_pycharm_project_989/data/dataset/
```

Plotting reliability curves:
```bash
# on mulga
scp $MY_USERNAME@$SLUG_SERVER:/home/$MY_USERNAME/tmp_pycharm_project_989/data/dataset/{robotreviewer_probs_low,rct_pdfs_out}* robotreviewer_temp/
# locally
scp $MY_USERNAME@$MULGA_SERVER:/silo-q04/users/s/$MY_USERNAME/robotreviewer_temp/{robotreviewer_probs_low,rct_pdfs_out}* /home/simon/Apps/SysRev/data/modelling/plots/robotreviewer/
```

Then look at `/home/simon/Apps/SysRev/sysrev/modelling/plots/reliability_curves_robotreviewer.R`.